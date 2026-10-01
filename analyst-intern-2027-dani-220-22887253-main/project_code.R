# NAME: Daniel Acevedo Vega | NUMBER: 2588
#
# project_code.R -- NBA shot make-probability model
#
# Run from the repo root:   Rscript project_code.R
#
# Fills submission.csv and writes the tables/figures that project_writeup.Rmd
# reads (results/, figures/). Nothing in the write-up refits a model.
#
# Pipeline
#   1. Feature engineering (geometry, defense, shooter context)
#   2. Out-of-fold shooter / defender skill with empirical-Bayes shrinkage
#   3. Models: logistic baseline -> XGBoost (context only) -> XGBoost + skill
#   4. Validation: leave-one-season-out (test is a season not seen in training)
#   5. Blend + calibration check, final fit on both seasons, predict test
#   6. Explainability: SHAP, partial dependence, calibration plot

suppressPackageStartupMessages({
  library(data.table)   # needs R.utils to read .csv.gz
  library(xgboost)      # >= 3.0
  library(splines)
  library(ggplot2)
})

set.seed(2027)
NTHREAD <- max(1L, parallel::detectCores() - 1L)
dir.create("figures", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

logloss <- function(y, p, eps = 1e-15) {
  p <- pmin(pmax(p, eps), 1 - eps)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}


# 1. Load ---------------------------------------------------------------------

train <- fread("training.csv.gz")
test  <- fread("testing.csv.gz")
train[, outcome := as.integer(outcome)]

cat("train seasons:", paste(unique(train$season_id), collapse = ", "),
    "| test seasons:", paste(unique(test$season_id), collapse = ", "), "\n")
# The test set is a third season that appears in neither training season, so
# every validation below holds out a whole season rather than random rows.


# 2. Feature engineering --------------------------------------------------------

# Hoop centre in the court coordinates. Checked on the training data:
# sqrt((locationx + 41.75)^2 + locationy^2) reproduces `distance` to ~1e-10.
HOOP_X <- -41.75

MONTHS <- c("OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR")
SHOTTYPES <- c("jumper", "layup", "dunk", "post", "floater", "tip", "heave")
GAMESTATES <- c("HALFCOURT", "TRANSITION", "DIRECTSECONDCHANCE")

shot_family <- function(shottype, three) {
  fifelse(three, "three",
          fifelse(shottype %in% c("dunk", "layup", "tip"), "rim", "mid"))
}

# closestdefapproach is a string like "{d_1.00s,d_0.75s,d_0.50s,d_0.25s}"
parse_approach <- function(s) {
  parts <- tstrsplit(gsub("[{}]", "", s), ",", fixed = TRUE)
  lapply(parts, as.numeric)
}

build_features <- function(d) {
  d <- copy(d)

  # Location
  d[, x_from_hoop := locationx - HOOP_X]
  d[, abs_y := abs(locationy)]
  d[, angle := atan2(abs_y, x_from_hoop) * 180 / pi]   # 0 = straight on, 90 = baseline
  d[, corner3 := as.integer(three & abs_y >= 21.5 & x_from_hoop <= 9.5)]
  d[, heave := as.integer(shottype == "heave" | distance >= 40)]
  d[, family := shot_family(shottype, three)]

  # Defense
  ap <- parse_approach(d$closestdefapproach)
  d[, `:=`(def_d100 = ap[[1]], def_d050 = ap[[3]], def_d025 = ap[[4]])]
  # Feet the closest defender closed in the final second / final quarter-second.
  # Positive = defender was getting closer at the release.
  d[, close_1s  := def_d100 - closestdefdist]
  d[, close_025 := def_d025 - closestdefdist]
  dc <- as.matrix(d[, .(distcont1, distcont2, distcont3, distcont4)])
  d[, contest_mean := fifelse(num_contesters > 0, rowMeans(dc, na.rm = TRUE), NA_real_)]
  d[, contest_pressure := rowSums(1 / pmax(dc, 0.5), na.rm = TRUE)]  # crowding: sum of 1/dist

  # Shooter context. Exactly-0 and exactly-24 shot clocks are over-represented
  # (~10k each) and behave like "clock off / just reset" codes, so flag them.
  d[, sc_zero := as.integer(shotclock == 0)]
  d[, sc_full := as.integer(shotclock == 24)]
  d[, late_clock := as.integer(shotclock > 0 & shotclock <= 4)]
  d[, month_num := match(month, MONTHS)]
  for (s in SHOTTYPES)  set(d, j = paste0("st_", s), value = as.integer(d$shottype == s))
  for (g in GAMESTATES) set(d, j = paste0("gs_", tolower(g)), value = as.integer(d$gamestate == g))
  d[, three := as.integer(three)]
  d
}

CONTEXT_FEATURES <- c(
  "distance", "x_from_hoop", "abs_y", "angle", "three", "corner3", "heave",
  paste0("st_", SHOTTYPES),
  "closestdefdist", "num_contesters", "contest_mean", "contest_pressure",
  "close_1s", "close_025", "def_d050",
  "shooterspeed", "dribblesbefore", "shotclock", "sc_zero", "sc_full", "late_clock",
  paste0("gs_", tolower(GAMESTATES)), "month_num"
)
SKILL_FEATURES <- c(
  "sk_fam", "sk_rim", "sk_mid", "sk_three",          # shooter makes-above-expected
  "def_skill"                                         # closest defender FG%-allowed vs expected
)
# Tried and dropped (see ablation below): shooter volume and 3PA rate. Inside
# training they are counted on 4/5 of the rows (out-of-fold) but on all rows
# for the scored season, so their scale shifts and the model overfits to them.
VOLUME_FEATURES <- c("n_fam_log", "n_all_log", "three_rate")

train <- build_features(train)
test  <- build_features(test)


# 3. Models -----------------------------------------------------------------------

XGB_PARAMS <- list(
  objective = "binary:logistic", eval_metric = "logloss", tree_method = "hist",
  eta = 0.03, max_depth = 6, min_child_weight = 100, subsample = 0.8,
  colsample_bytree = 0.7, lambda = 5, nthread = NTHREAD
)
BASE_PARAMS <- modifyList(XGB_PARAMS, list(eta = 0.08))
BASE_ROUNDS <- 400

xgb_fit <- function(X, y, nrounds, params = XGB_PARAMS, Xval = NULL, yval = NULL) {
  dtr <- xgb.DMatrix(X, label = y)
  evals <- if (!is.null(Xval)) list(val = xgb.DMatrix(Xval, label = yval)) else list()
  xgb.train(params = params, data = dtr, nrounds = nrounds, evals = evals, verbose = 0,
            early_stopping_rounds = if (length(evals)) 150 else NULL)
}

xgb_pred <- function(model, X, n = NULL) {
  if (is.null(n)) predict(model, X) else predict(model, X, iterationrange = c(1, n))
}

logit_fit <- function(d) {
  glm(outcome ~ ns(distance, 5) + shottype + three + ns(closestdefdist, 4),
      data = d, family = binomial())
}


# 3a. Shooter / defender skill ------------------------------------------------------
#
# Skill = a player's makes above *expected*, where "expected" is an out-of-fold
# context-only model (no player identity). Per shooter and shot family:
#     raw   = sum(outcome - p_hat) / n
#     shrunk = sum(outcome - p_hat) / (n + k),   k = sigma^2 / tau^2
# tau^2 (true between-player variance) is estimated by method of moments, so
# low-volume shooters are pulled toward 0 (league average given context).
#
# Leakage control: a training row never sees its own outcome in either p_hat or
# its skill value (both are out-of-fold). Rows being predicted (validation
# season or test) get skill computed from all training rows, mirroring the
# real test situation.

skill_table <- function(res, id_col, fam_col, shrink = TRUE) {
  s <- res[, .(r = sum(outcome - p_hat), n = .N, v = sum(p_hat * (1 - p_hat))),
           by = c(id_col, fam_col)]
  setnames(s, c(id_col, fam_col), c("id", "fam"))
  if (shrink) {
    s[, k := {
      big   <- n >= 50
      noise <- mean(v[big] / n[big]^2)                 # sampling var of a player's mean residual
      tau2  <- max(var((r / n)[big]) - noise, 1e-5)    # true skill variance
      mean(v[big] / n[big]) / tau2
    }, by = fam]
  } else {
    s[, k := 0]
  }
  s[, skill := r / (n + k)]
  s[]
}

attach_skill <- function(d, res, shrink = TRUE) {
  d <- copy(d)
  sh <- skill_table(res, "shooter_id", "family", shrink)
  wide <- dcast(sh, id ~ fam, value.var = c("skill", "n"), fill = 0)
  setnames(wide, sub("^skill_", "sk_", names(wide)))
  for (f in c("rim", "mid", "three")) {
    for (pfx in c("sk_", "n_")) if (!paste0(pfx, f) %in% names(wide)) wide[, (paste0(pfx, f)) := 0]
  }
  d[wide, on = .(shooter_id = id),
    `:=`(sk_rim = i.sk_rim, sk_mid = i.sk_mid, sk_three = i.sk_three,
         n_rim = i.n_rim, n_mid = i.n_mid, n_three = i.n_three)]
  for (col in c("sk_rim", "sk_mid", "sk_three", "n_rim", "n_mid", "n_three")) {
    set(d, which(is.na(d[[col]])), col, 0)                 # unseen shooter = league average
  }
  d[, sk_fam := fcase(family == "rim", sk_rim, family == "mid", sk_mid, default = sk_three)]
  d[, n_fam := fcase(family == "rim", n_rim, family == "mid", n_mid, default = n_three)]
  d[, n_all := n_rim + n_mid + n_three]
  d[, `:=`(n_fam_log = log1p(n_fam), n_all_log = log1p(n_all),
           three_rate = fifelse(n_all > 0, n_three / n_all, NA_real_))]

  res2 <- copy(res)[, all := "all"]
  df <- skill_table(res2, "closestdef_id", "all", shrink)
  d[df, on = .(closestdef_id = id), def_skill := i.skill]
  d[is.na(def_skill), def_skill := 0]
  d[, c("n_fam", "n_all", "n_rim", "n_mid", "n_three") := NULL]
  d
}

# Out-of-fold context-only predictions on a training set (the "expected" in
# makes-above-expected).
oof_context <- function(tr, k = 5) {
  fold <- sample(rep_len(1:k, nrow(tr)))
  p <- numeric(nrow(tr))
  X <- as.matrix(tr[, ..CONTEXT_FEATURES])
  for (f in 1:k) {
    m <- xgb_fit(X[fold != f, ], tr$outcome[fold != f], BASE_ROUNDS, BASE_PARAMS)
    p[fold == f] <- xgb_pred(m, X[fold == f, ])
  }
  list(p_hat = p, fold = fold)
}

# Add skill features to a training set (out-of-fold) and a target set (from all
# of training).
add_skill_features <- function(tr, tg, base, shrink = TRUE) {
  res <- tr[, .(shooter_id, closestdef_id, family, outcome)][, p_hat := base$p_hat]
  tr_out <- vector("list", max(base$fold))
  for (f in seq_along(tr_out)) {
    tr_out[[f]] <- attach_skill(tr[base$fold == f], res[base$fold != f], shrink)
  }
  tr_sk <- rbindlist(tr_out)
  tr_sk <- tr_sk[match(tr$shot_id, tr_sk$shot_id)]
  list(tr = tr_sk, tg = attach_skill(tg, res, shrink))
}


# 4. Leave-one-season-out validation ----------------------------------------------

seasons <- unique(train$season_id)
MAX_ROUNDS <- 3000

best_n <- function(model) which.min(attributes(model)$evaluation_log$val_logloss)

val_rows <- list()
best_iters <- list()

for (s in seasons) {
  cat("\n== Holding out season", s, "==\n")
  tr <- train[season_id != s]
  va <- train[season_id == s]
  y  <- va$outcome

  preds <- list()
  preds$logistic <- predict(logit_fit(tr), va, type = "response")

  Xtr <- as.matrix(tr[, ..CONTEXT_FEATURES]); Xva <- as.matrix(va[, ..CONTEXT_FEATURES])
  m <- xgb_fit(Xtr, tr$outcome, MAX_ROUNDS, Xval = Xva, yval = y)
  best_iters[[paste(s, "context")]] <- best_n(m)
  preds$xgb_context <- xgb_pred(m, Xva, best_n(m))

  base <- oof_context(tr)
  feats <- c(CONTEXT_FEATURES, SKILL_FEATURES)
  for (mode in c("raw", "shrunk", "volume")) {
    sk <- add_skill_features(tr, va, base, shrink = (mode != "raw"))
    f <- if (mode == "volume") c(feats, VOLUME_FEATURES) else feats
    Xtr <- as.matrix(sk$tr[, ..f]); Xva <- as.matrix(sk$tg[, ..f])
    m <- xgb_fit(Xtr, tr$outcome, MAX_ROUNDS, Xval = Xva, yval = y)
    best_iters[[paste(s, mode)]] <- best_n(m)
    preds[[paste0("xgb_skill_", mode)]] <- xgb_pred(m, Xva, best_n(m))
    cat(mode, "skill: best rounds", best_n(m), "\n")
  }

  # Leakage demo: shrunk skill computed *in-sample* (each row's own outcome
  # included). Looks great on the training rows, hurts on the held-out season.
  res_in <- tr[, .(shooter_id, closestdef_id, family, outcome)][, p_hat := base$p_hat]
  tr_leak <- attach_skill(tr, res_in, TRUE)
  va_leak <- attach_skill(va, res_in, TRUE)
  m <- xgb_fit(as.matrix(tr_leak[, ..feats]), tr$outcome, MAX_ROUNDS,
               Xval = as.matrix(va_leak[, ..feats]), yval = y)
  preds$xgb_skill_leaky <- xgb_pred(m, as.matrix(va_leak[, ..feats]), best_n(m))

  val_rows[[s]] <- data.table(season = s, shot_id = va$shot_id, outcome = y,
                              shottype = va$shottype, family = va$family, as.data.table(preds))
  print(sapply(preds, logloss, y = y))
}

val <- rbindlist(val_rows)


# 5. Blend + calibration ----------------------------------------------------------

# Blend in log-odds space; weight chosen on the pooled held-out predictions.
blend <- function(p1, p2, w) plogis(w * qlogis(p1) + (1 - w) * qlogis(p2))
ws <- seq(0, 1, by = 0.05)
blend_ll <- sapply(ws, function(w) logloss(val$outcome, blend(val$xgb_skill_shrunk, val$logistic, w)))
W_BLEND <- ws[which.min(blend_ll)]
val[, blend := blend(xgb_skill_shrunk, logistic, W_BLEND)]
cat("\nBlend weight on XGBoost:", W_BLEND, "\n")

# Calibration, cross-fitted: fit the calibrator on one held-out season, apply to
# the other, so the reported gain is honest.
val[, `:=`(platt = NA_real_, isotonic = NA_real_)]
for (s in seasons) {
  fit_rows <- val$season != s; app_rows <- val$season == s
  pl <- glm(outcome ~ qlogis(blend), data = val[fit_rows], family = binomial())
  val[app_rows, platt := predict(pl, val[app_rows], type = "response")]
  iso <- with(val[fit_rows][order(blend)], isoreg(blend, outcome))
  iso_fun <- approxfun(sort(val[fit_rows]$blend), iso$yf, rule = 2, ties = mean)
  val[app_rows, isotonic := pmin(pmax(iso_fun(blend), 1e-4), 1 - 1e-4)]
}

model_labels <- c(
  logistic = "1. Logistic baseline (distance, shot type, three, defender dist.)",
  xgb_context = "2. XGBoost, context features only",
  xgb_skill_leaky = "x. XGBoost + skill computed in-sample (leaky)",
  xgb_skill_raw = "3. XGBoost + raw (unshrunk) out-of-fold skill",
  xgb_skill_volume = "x. XGBoost + shrunk skill + shooter volume/3PA rate",
  xgb_skill_shrunk = "4. XGBoost + shrunk out-of-fold skill",
  blend = sprintf("5. Blend (%.2f XGBoost / %.2f logistic)", W_BLEND, 1 - W_BLEND),
  platt = "6a. Blend + Platt scaling (cross-fitted)",
  isotonic = "6b. Blend + isotonic (cross-fitted)"
)
metrics <- rbindlist(lapply(names(model_labels), function(mname) {
  row <- data.table(model = model_labels[[mname]], id = mname,
                    pooled = logloss(val$outcome, val[[mname]]))
  for (s in seasons) row[, (paste0("holdout_", s)) := logloss(val[season == s]$outcome, val[season == s][[mname]])]
  row
}))
metrics[, null_model := logloss(val$outcome, rep(mean(train$outcome), nrow(val)))]
fwrite(metrics, "results/model_logloss.csv")
print(metrics)

# Decide whether a calibrator earns its place (needs a real gain, not noise).
cal_gain <- metrics[id == "blend", pooled] - metrics[id %in% c("platt", "isotonic"), min(pooled)]
USE_CAL <- if (cal_gain > 5e-4) metrics[id %in% c("platt", "isotonic")][which.min(pooled), id] else "none"
cat("Calibration choice:", USE_CAL, sprintf("(gain %.5f)\n", cal_gain))

# Rounds for the final fit: mean best iteration across folds, scaled up a bit
# because the final model sees twice as much data.
N_FINAL <- round(1.15 * mean(unlist(best_iters[grepl("shrunk", names(best_iters))])))
cat("Final boosting rounds:", N_FINAL, "\n")


# 6. Final fit on both seasons, predict test ----------------------------------------

cat("\n== Final fit ==\n")
base_all <- oof_context(train)
sk_all <- add_skill_features(train, test, base_all, shrink = TRUE)
FEATURES <- c(CONTEXT_FEATURES, SKILL_FEATURES)
X_all  <- as.matrix(sk_all$tr[, ..FEATURES])
X_test <- as.matrix(sk_all$tg[, ..FEATURES])
final_xgb <- xgb_fit(X_all, train$outcome, N_FINAL)
final_logit <- logit_fit(train)

p_test <- blend(xgb_pred(final_xgb, X_test), predict(final_logit, test, type = "response"), W_BLEND)
if (USE_CAL == "platt") {
  pl <- glm(outcome ~ qlogis(blend), data = val, family = binomial())
  p_test <- predict(pl, data.frame(blend = p_test), type = "response")
} else if (USE_CAL == "isotonic") {
  iso <- with(val[order(blend)], isoreg(blend, outcome))
  p_test <- approxfun(sort(val$blend), iso$yf, rule = 2, ties = mean)(p_test)
}
p_test <- pmin(pmax(p_test, 1e-4), 1 - 1e-4)

sub <- fread("submission.csv")
stopifnot(identical(as.character(sub$shot_id), as.character(test$shot_id)))
sub[, make_prob := p_test]
fwrite(sub, "submission.csv")
cat("submission.csv written:", nrow(sub), "rows, mean make_prob",
    round(mean(p_test), 4), "(train make rate", round(mean(train$outcome), 4), ")\n")
xgb.save(final_xgb, "results/final_xgb.ubj")


# 7. Explainability -----------------------------------------------------------------

theme_set(theme_minimal(base_size = 11))
BLUE <- "#2a6fdb"

feature_labels <- c(
  distance = "Shot distance", x_from_hoop = "Depth from hoop (x)", abs_y = "Lateral offset |y|",
  angle = "Shot angle", three = "Three-pointer", corner3 = "Corner three", heave = "Heave",
  st_jumper = "Type: jumper", st_layup = "Type: layup", st_dunk = "Type: dunk", st_post = "Type: post",
  st_floater = "Type: floater", st_tip = "Type: tip", st_heave = "Type: heave",
  closestdefdist = "Closest defender dist.", num_contesters = "# contesters",
  contest_mean = "Mean contester dist.", contest_pressure = "Contest pressure (sum 1/dist)",
  close_1s = "Defender closing (1s)", close_025 = "Defender closing (0.25s)",
  def_d050 = "Defender dist. 0.5s before", shooterspeed = "Shooter speed",
  dribblesbefore = "Dribbles before", shotclock = "Shot clock", sc_zero = "Shot clock = 0",
  sc_full = "Shot clock = 24", late_clock = "Late clock (<=4s)", gs_halfcourt = "Halfcourt",
  gs_transition = "Transition", gs_directsecondchance = "Second chance", month_num = "Month",
  sk_fam = "Shooter skill (this shot family)", sk_rim = "Shooter skill: rim",
  sk_mid = "Shooter skill: midrange", sk_three = "Shooter skill: three",
  n_fam_log = "Shooter volume (this family)", n_all_log = "Shooter volume (all)",
  three_rate = "Shooter 3PA rate", def_skill = "Closest defender skill"
)

# SHAP on a random sample of test shots
idx <- sample(nrow(X_test), 20000)
shap <- predict(final_xgb, X_test[idx, ], predcontrib = TRUE)
shap <- shap[, colnames(shap) != "BIAS" & colnames(shap) != "(Intercept)", drop = FALSE]
colnames(shap) <- FEATURES
imp <- data.table(feature = FEATURES, mean_abs_shap = colMeans(abs(shap)))[order(-mean_abs_shap)]
imp[, label := feature_labels[feature]]
fwrite(imp, "results/shap_importance.csv")

top <- imp[1:20]
g <- ggplot(top, aes(x = mean_abs_shap, y = reorder(label, mean_abs_shap))) +
  geom_col(fill = BLUE, width = 0.7) +
  labs(x = "Mean |SHAP value| (log-odds)", y = NULL,
       title = "What the model leans on", subtitle = "Top 20 features, 20,000 test shots") +
  theme(panel.grid.major.y = element_blank())
ggsave("figures/shap_importance.png", g, width = 7, height = 5.5, dpi = 150)

# SHAP dependence for a few key features
dep_feats <- c("distance", "closestdefdist", "close_1s", "sk_fam", "shotclock", "shooterspeed")
dep <- rbindlist(lapply(dep_feats, function(f) data.table(
  feature = feature_labels[[f]], value = X_test[idx, f], shap = shap[, f])))
dep <- dep[, .SD[value >= quantile(value, .005, na.rm = TRUE) & value <= quantile(value, .995, na.rm = TRUE)],
           by = feature]
g <- ggplot(dep, aes(value, shap)) +
  geom_point(alpha = 0.06, size = 0.5, colour = BLUE) +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), colour = "black", linewidth = 0.6, se = FALSE) +
  geom_hline(yintercept = 0, linetype = 2, colour = "grey50") +
  facet_wrap(~feature, scales = "free_x", ncol = 3) +
  labs(x = NULL, y = "SHAP value (log-odds)", title = "SHAP dependence: how each feature moves a shot's odds")
ggsave("figures/shap_dependence.png", g, width = 9, height = 5.5, dpi = 150)

# Partial dependence on jumpers (average prediction as one feature is swept)
jump_idx <- sample(which(test$shottype == "jumper"), 5000)
Xj <- X_test[jump_idx, ]
pdp <- function(X, feature, grid, mutate = NULL) {
  rbindlist(lapply(grid, function(v) {
    Xm <- X; Xm[, feature] <- v
    if (!is.null(mutate)) Xm <- mutate(Xm, v)
    data.table(value = v, make_prob = mean(predict(final_xgb, Xm)))
  }))
}

# Distance: slide each jumper along its own ray from the hoop, keeping the angle,
# and recompute the location features consistently.
move_along_ray <- function(Xm, v) {
  th <- Xm[, "angle"] * pi / 180
  Xm[, "x_from_hoop"] <- v * cos(th); Xm[, "abs_y"] <- v * sin(th)
  is3 <- as.integer(v > 23.75 | (Xm[, "abs_y"] >= 22 & Xm[, "x_from_hoop"] <= 9.5))
  Xm[, "three"] <- is3
  Xm[, "corner3"] <- as.integer(is3 == 1 & Xm[, "abs_y"] >= 21.5 & Xm[, "x_from_hoop"] <= 9.5)
  Xm[, "sk_fam"] <- ifelse(is3 == 1, Xm[, "sk_three"], Xm[, "sk_mid"])
  Xm
}
# The defense features are strongly correlated (distance at release, 0.5s
# before, contester mean distance, contest pressure), so sweeping one column
# alone lets the model attribute the effect to the frozen others. Instead move
# the whole defender consistently.
#   Defender distance: shift the closest defender to v ft, keeping their approach
#   path (how much they closed) and any other contesters unchanged.
move_defender <- function(Xm, v) {
  old <- X_orig_def
  shift <- v - old[, "closestdefdist"]
  Xm[, "def_d050"] <- old[, "def_d050"] + shift
  contested <- old[, "num_contesters"] > 0
  Xm[contested, "contest_mean"] <- old[contested, "contest_mean"] + shift[contested] / old[contested, "num_contesters"]
  Xm[contested, "contest_pressure"] <- old[contested, "contest_pressure"] -
    1 / pmax(old[contested, "closestdefdist"], 0.5) + 1 / max(v, 0.5)
  Xm
}
#   Closing speed: defender approaches linearly at v ft/s and ends at their real
#   release distance, so d(t before release) = release distance + v * t.
set_closing <- function(Xm, v) {
  Xm[, "close_025"] <- v * 0.25
  Xm[, "def_d050"] <- Xm[, "closestdefdist"] + v * 0.5
  Xm
}
X_orig_def <- Xj
pd_dist  <- pdp(Xj, "distance", seq(8, 30, by = 1), move_along_ray)
pd_def   <- pdp(Xj, "closestdefdist", seq(1, 12, by = 0.5), move_defender)
pd_close <- pdp(Xj, "close_1s", seq(-2, 8, by = 0.5), set_closing)
pd_all <- rbind(
  pd_dist[, feature := "Shot distance (ft)"],
  pd_def[, feature := "Closest defender distance at release (ft)"],
  pd_close[, feature := "Defender closing speed: ft closed in last 1s"])
fwrite(pd_all, "results/partial_dependence_jumpers.csv")
g <- ggplot(pd_all, aes(value, make_prob)) +
  geom_line(colour = BLUE, linewidth = 1) + geom_point(colour = BLUE, size = 1.2) +
  facet_wrap(~feature, scales = "free_x") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(x = NULL, y = "Average predicted make %", title = "Partial dependence on 5,000 test jumpers") +
  theme(panel.spacing = unit(1.5, "lines"))
ggsave("figures/partial_dependence.png", g, width = 10, height = 3.6, dpi = 150)

# Calibration plot on the held-out seasons
cal_models <- c("logistic", "xgb_context", "xgb_skill_shrunk", if (W_BLEND < 1) "blend")
cal <- rbindlist(lapply(cal_models, function(mname) {
  val[, .(pred = mean(get(mname)), actual = mean(outcome), n = .N),
      by = .(bin = cut(get(mname), breaks = seq(0, 1, 0.05), include.lowest = TRUE))][
    , model := model_labels[[mname]]]
}))
fwrite(cal, "results/calibration_bins.csv")
g <- ggplot(cal[n >= 200], aes(pred, actual, colour = model)) +
  geom_abline(linetype = 2, colour = "grey50") +
  geom_line() + geom_point(aes(size = n), alpha = 0.8) +
  scale_size_area(max_size = 4, guide = "none") +
  scale_colour_manual(values = c("#9aa5b1", "#e8a33d", BLUE, "#1b1f24")[seq_along(cal_models)]) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "Predicted make probability (bin mean)", y = "Actual make rate", colour = NULL,
       title = "Calibration on held-out seasons") +
  theme(legend.position = "bottom", legend.direction = "vertical")
ggsave("figures/calibration.png", g, width = 6, height = 7, dpi = 150)

# Skill shrinkage illustration (three-point family, full training data)
res_all <- train[, .(shooter_id, closestdef_id, family, outcome)][, p_hat := base_all$p_hat]
sh_raw <- skill_table(res_all, "shooter_id", "family", shrink = FALSE)
sh_eb  <- skill_table(res_all, "shooter_id", "family", shrink = TRUE)
shr <- sh_raw[, .(id, fam, n, raw = skill)][sh_eb[, .(id, fam, shrunk = skill, k)], on = .(id, fam)]
fwrite(shr[, .(k = k[1], shooters = .N), by = fam], "results/shrinkage_k.csv")
g <- ggplot(shr[fam == "three"], aes(n)) +
  geom_point(aes(y = raw), colour = "#9aa5b1", alpha = 0.5, size = 0.8) +
  geom_point(aes(y = shrunk), colour = BLUE, alpha = 0.7, size = 0.8) +
  scale_x_log10() + scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  coord_cartesian(ylim = c(-0.4, 0.4)) +
  labs(x = "Three-point attempts in training (log scale)", y = "Makes above expected per shot",
       title = "Empirical-Bayes shrinkage of three-point skill",
       subtitle = "Grey = raw, blue = shrunk. Small samples get pulled to 0.")
ggsave("figures/shrinkage.png", g, width = 7, height = 4.2, dpi = 150)

# Logistic baseline coefficients (for the write-up)
fwrite(as.data.table(summary(final_logit)$coefficients, keep.rownames = "term"),
       "results/logistic_coefficients.csv")
saveRDS(list(W_BLEND = W_BLEND, USE_CAL = USE_CAL, N_FINAL = N_FINAL, best_iters = best_iters,
             seasons = seasons, test_season = unique(test$season_id)), "results/run_info.rds")

cat("\nDone.\n")
