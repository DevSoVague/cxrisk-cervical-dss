# Cervical Cancer DSS Prototype (Shiny)
# Clinician-facing cervical cancer screening triage prototype.
#
# Two run modes, chosen from environment variables at startup:
#
#   DEMO MODE (SUPABASE_HOST unset)
#     No login, no database. Patients are synthetic demo records built from the
#     public UCI CSV in ../data/. Changes (new patients, overrides) live in memory
#     for the session only. Scoring goes to the FastAPI service at
#     CERVICAL_API_URL (default http://localhost:8001/predict) and falls back to
#     the local logistic model if the API is unreachable.
#
#   DB MODE (SUPABASE_HOST set)
#     shinymanager login against the app_user table (bcrypt hashes), patients
#     loaded from and saved to Postgres (see schema.sql). Needs SUPABASE_HOST,
#     SUPABASE_USER and SUPABASE_PASSWORD; optional SUPABASE_DBNAME (default
#     "postgres"), SUPABASE_PORT (default 6543) and SUPABASE_SSLMODE (default
#     "require"). Scoring uses the FastAPI
#     service only when CERVICAL_API_URL is set.
#
# Optional for both: CERVICAL_API_KEY (sent as a Bearer token),
# CERVICAL_API_TIMEOUT (seconds, default 10).
#
# Run from the repository root:  Rscript -e "shiny::runApp('frontend')"

library(shiny)
library(DT)
library(dplyr)
library(ggplot2)
library(scales)
library(httr)
library(jsonlite)

DEMO_MODE <- !nzchar(Sys.getenv("SUPABASE_HOST", unset = ""))

if (!DEMO_MODE) {
  library(shinymanager)
  library(DBI)
  library(RPostgres)
  library(pool)
  library(bcrypt)
}

# custom null coalescing operator 
`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0) a else b

# ---------------------------
# Helpers
# ---------------------------
# convert column names to R safe variable names 
clean_names <- function(x) {
  x <- trimws(x)
  x <- gsub("Smokes \\(packs/year\\)", "Smokes packs per year", x)
  x <- gsub("First sexual intercourse", "First sexual intercourse age", x)
  x <- gsub("STDs \\(number\\)", "STDs number", x)
  x <- gsub("Hormonal Contraceptives \\(years\\)", "Hormonal Contraceptives years", x)
  x <- gsub("IUD \\(years\\)", "IUD years", x)
  x <- gsub("Smokes \\(years\\)", "Smokes years", x)
  x <- gsub(":", " ", x)
  x <- gsub("[()]", " ", x)
  x <- gsub("/", " per ", x)
  x <- gsub("[^0-9A-Za-z]+", "_", x)
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  tolower(x)
}

# replaces NA values with fallback value 
coalesce_num <- function(x, fallback = 0) {
  x <- suppressWarnings(as.numeric(x))
  ifelse(is.na(x), fallback, x)
}

# convert 1/0 to "Yes"/"No" 
binary_label <- function(x) ifelse(as.numeric(x) >= 1, "Yes", "No")

# format 10 digit phone number 
format_phone <- function(x) {
  if (is.null(x) || is.na(x) || !nzchar(trimws(x))) return("")  # add this
  digits <- gsub("[^0-9]", "", x)
  if (nchar(digits) == 11 && startsWith(digits, "1")) {
    digits <- substr(digits, 2, 11)
  }
  if (nchar(digits) == 10) {
    paste0("(", substr(digits, 1, 3), ") ",
           substr(digits, 4, 6), "-",
           substr(digits, 7, 10))
  } else {
    x
  }
}

# ---------------------------
# Data loading (public UCI CSV shipped in ../data/)
# ---------------------------
# The cleaned table drops exact duplicate rows and keeps NA; the analysis table
# median-imputes every numeric column. Both feed the local fallback model and
# the population dashboard; demo-mode patients are drawn from the cleaned table.
UCI_CSV_CANDIDATES <- c(
  Sys.getenv("CXRISK_DATA_CSV", unset = ""),
  file.path("..", "data", "risk_factors_cervical_cancer.csv"),
  file.path("data", "risk_factors_cervical_cancer.csv")
)

load_project_data <- function() {
  csv_path <- UCI_CSV_CANDIDATES[nzchar(UCI_CSV_CANDIDATES) & file.exists(UCI_CSV_CANDIDATES)][1]
  if (is.na(csv_path)) {
    stop(
      "UCI data file not found. Run the app from the repository root with ",
      "shiny::runApp('frontend'), or set CXRISK_DATA_CSV to the CSV path.",
      call. = FALSE
    )
  }

  raw_df <- read.csv(
    csv_path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    na.strings = c("?", "")
  )
  names(raw_df) <- clean_names(names(raw_df))

  clean_df <- dplyr::distinct(raw_df)

  analysis_df <- clean_df
  for (nm in names(analysis_df)) {
    x <- suppressWarnings(as.numeric(analysis_df[[nm]]))
    med <- stats::median(x, na.rm = TRUE)
    x[is.na(x)] <- ifelse(is.na(med), 0, med)
    analysis_df[[nm]] <- x
  }

  outcome_cols_clean <- intersect(c("hinselmann", "schiller", "citology", "biopsy"), names(clean_df))
  outcome_cols_analysis <- intersect(c("hinselmann", "schiller", "citology", "biopsy"), names(analysis_df))
  
  if (!"abnormal_any" %in% names(clean_df) && length(outcome_cols_clean) > 0) {
    clean_df$abnormal_any <- do.call(pmax, c(clean_df[outcome_cols_clean], na.rm = TRUE))
  }
  
  if (!"missing_count_predictors" %in% names(clean_df) && length(outcome_cols_clean) > 0) {
    pred_cols <- setdiff(names(clean_df), outcome_cols_clean)
    clean_df$missing_count_predictors <- rowSums(is.na(clean_df[pred_cols]))
  }
  
  if (!"abnormal_any" %in% names(analysis_df) && length(outcome_cols_analysis) > 0) {
    analysis_df$abnormal_any <- do.call(pmax, c(analysis_df[outcome_cols_analysis], na.rm = TRUE))
  }
  
  list(raw = raw_df, clean = clean_df, analysis = analysis_df)
}

project_data <- load_project_data()
raw_df <- project_data$raw
clean_df <- project_data$clean
analysis_df <- project_data$analysis
patients_df <- analysis_df

# ---------------------------
# Feature engineering for the prototype model
# ---------------------------
feature_pool <- c(
  "age",
  "number_of_sexual_partners",
  "first_sexual_intercourse_age",
  "num_of_pregnancies",
  "smokes",
  "smokes_years",
  "smokes_packs_per_year",
  "hormonal_contraceptives",
  "hormonal_contraceptives_years",
  "iud",
  "iud_years",
  "stds",
  "stds_number",
  "dx_cancer",
  "dx_cin",
  "dx_hpv",
  "dx"
)

available_features <- intersect(feature_pool, names(analysis_df))
analysis_model <- analysis_df

for (nm in available_features) {
  analysis_model[[nm]] <- suppressWarnings(as.numeric(analysis_model[[nm]]))
}

# computes median of features for future imputation 
feature_medians <- sapply(analysis_model[, available_features, drop = FALSE], function(x) {
  x <- suppressWarnings(as.numeric(x))
  med <- stats::median(x, na.rm = TRUE)
  ifelse(is.na(med), 0, med)
})

feature_labels <- c(
  age = "Age",
  number_of_sexual_partners = "Number of sexual partners",
  first_sexual_intercourse_age = "Age at first sexual intercourse",
  num_of_pregnancies = "Number of pregnancies",
  smokes = "Smoking status",
  smokes_years = "Smoking years",
  smokes_packs_per_year = "Smoking packs per year",
  hormonal_contraceptives = "Hormonal contraceptive use",
  hormonal_contraceptives_years = "Hormonal contraceptive years",
  iud = "IUD use",
  iud_years = "IUD years",
  stds = "Any STD history",
  stds_number = "Number of STDs",
  dx_cancer = "Prior cancer diagnosis",
  dx_cin = "Prior CIN diagnosis",
  dx_hpv = "Prior HPV diagnosis",
  dx = "Any prior diagnosis flag"
)

# logistic regression model predicting abnormal_any from all features 
glm_formula <- as.formula(
  paste("abnormal_any ~", paste(available_features, collapse = " + "))
)

risk_model <- tryCatch({
  stats::glm(glm_formula, data = analysis_model, family = binomial())
}, warning = function(w) {
  suppressWarnings(stats::glm(glm_formula, data = analysis_model, family = binomial()))
}, error = function(e) NULL)

score_patient <- function(patient_row, low_thr = 0.15, high_thr = 0.35) {
  row <- patient_row[1, , drop = FALSE]
  
  for (nm in available_features) {
    if (!nm %in% names(row)) row[[nm]] <- feature_medians[[nm]]
    row[[nm]] <- coalesce_num(row[[nm]], feature_medians[[nm]])
  }
  
  if (!is.null(risk_model)) {
    prob <- as.numeric(stats::predict(risk_model, newdata = row, type = "response"))
    prob <- max(min(prob, 0.999), 0.001)
    
    coefs <- stats::coef(risk_model)
    coefs <- coefs[setdiff(names(coefs), "(Intercept)")]
    usable_coef_names <- intersect(names(coefs), available_features)
    vals <- sapply(usable_coef_names, function(nm) coalesce_num(row[[nm]], feature_medians[[nm]])[1])
    meds <- feature_medians[usable_coef_names]
    contrib <- coefs[usable_coef_names] * (vals - meds)
    
    top <- sort(contrib[contrib > 0], decreasing = TRUE)
    if (length(top) < 3) {
      top <- sort(abs(contrib), decreasing = TRUE)
    }
    driver_names <- names(utils::head(top, 3))
  } else {
    # Simple fallback if model cannot be fit
    linear_score <-
      0.02 * coalesce_num(row$age, 35) +
      0.25 * coalesce_num(row$smokes, 0) +
      0.20 * coalesce_num(row$stds, 0) +
      0.30 * coalesce_num(row$dx_hpv, 0) +
      0.20 * coalesce_num(row$dx_cancer, 0) +
      0.06 * coalesce_num(row$num_of_pregnancies, 0) +
      0.04 * coalesce_num(row$hormonal_contraceptives_years, 0)
    prob <- 1 / (1 + exp(-(-4 + linear_score)))
    driver_names <- c("age", "smokes", "stds")
  }
  
  risk_level <- if (prob < low_thr) {
    "Low"
  } else if (prob < high_thr) {
    "Medium"
  } else {
    "High"
  }
  
  recommendation <- dplyr::case_when(
    risk_level == "Low" ~ "Routine recall",
    risk_level == "Medium" ~ "Expedited HPV/Pap testing",
    TRUE ~ "Referral for further evaluation"
  )
  
  list(
    probability = prob,
    risk_level = risk_level,
    recommendation = recommendation,
    top_drivers = unname(feature_labels[driver_names]),
    driver_code = driver_names
  )
}

# In demo mode the API defaults to a local uvicorn on port 8001; in DB mode it
# is used only when CERVICAL_API_URL is set explicitly.
DEFAULT_API_URL <- "http://localhost:8001/predict"

api_url <- function() {
  url <- Sys.getenv("CERVICAL_API_URL", unset = "")
  if (!nzchar(url) && DEMO_MODE) url <- DEFAULT_API_URL
  url
}

api_is_configured <- function() {
  nzchar(api_url())
}

build_api_payload <- function(patient_row) {
  row <- patient_row[1, , drop = FALSE]
  list(
    age             = unname(coalesce_num(row$age, feature_medians[["age"]])[1]),
    sexual_partners = unname(coalesce_num(row$number_of_sexual_partners, feature_medians[["number_of_sexual_partners"]])[1]),
    pregnancies     = unname(coalesce_num(row$num_of_pregnancies, feature_medians[["num_of_pregnancies"]])[1]),
    smokes_years    = unname(coalesce_num(row$smokes_years, 0)[1]),
    hc_years        = unname(coalesce_num(row$hormonal_contraceptives_years, 0)[1]),
    iud_years       = unname(coalesce_num(row$iud_years, 0)[1]),
    std_burden      = unname(coalesce_num(row$stds_number, coalesce_num(row$stds, 0))[1])
  )
}

normalize_api_response <- function(api_body, local_score) {
  prob <- suppressWarnings(as.numeric(
    api_body[["probability"]] %||%
      api_body[["predicted_probability"]] %||%
      api_body[["risk_score"]] %||%
      local_score$probability
  ))
  prob <- max(min(prob, 0.999), 0.001)
  
  raw_level <- as.character(
    api_body[["prediction"]] %||%
      api_body[["risk_level"]] %||%
      local_score$risk_level
  )
  risk_level <- if (grepl("HIGH", toupper(raw_level))) "High" else
    if (grepl("LOW",  toupper(raw_level))) "Low"  else
      if (grepl("MED",  toupper(raw_level))) "Medium" else
        local_score$risk_level
  
  recommendation <- as.character(
    api_body[["recommendation"]] %||%
      api_body[["recommended_action"]] %||%
      dplyr::case_when(
        risk_level == "Low"    ~ "Routine recall",
        risk_level == "Medium" ~ "Expedited HPV/Pap testing",
        TRUE                   ~ "Referral for further evaluation"
      )
  )
  
  top_drivers <- unlist(
    api_body[["top_drivers"]] %||%
      api_body[["drivers"]] %||%
      list(local_score$top_drivers)
  )
  top_drivers <- as.character(top_drivers[nzchar(top_drivers)])
  if (length(top_drivers) == 0) top_drivers <- local_score$top_drivers
  
  list(
    probability    = prob,
    risk_level     = risk_level,
    recommendation = recommendation,
    top_drivers    = top_drivers,
    driver_code    = local_score$driver_code,
    source         = "External API",
    api_status     = "API scoring completed successfully.",
    # Model Insights fields
    confidence        = suppressWarnings(as.numeric(api_body[["confidence"]] %||% NA)),
    confidence_tier   = as.character(api_body[["confidence_tier"]] %||% ""),
    nl_explanation    = as.character(api_body[["nl_explanation"]] %||% ""),
    warnings          = unlist(api_body[["warnings"]] %||% list()),
    model_probs       = api_body[["model_probs"]] %||% list(),
    scatter_png_b64      = as.character(api_body[["scatter_png_b64"]]      %||% ""),
    waterfall_png_b64    = as.character(api_body[["waterfall_png_b64"]]    %||% ""),
    contribution_png_b64 = as.character(api_body[["contribution_png_b64"]] %||% "")
  )
}

call_risk_api <- function(patient_row, low_thr = 0.15, high_thr = 0.35) {
  # always compute local score first, used as fallback
  local_score <- score_patient(patient_row, low_thr, high_thr)
  local_score$source     <- "Local model"
  local_score$api_status <- "Local model used."
  
  if (!api_is_configured()) return(local_score)
  
  result <- tryCatch({
    resp <- httr::POST(
      url     = api_url(),
      httr::add_headers(
        "Content-Type"  = "application/json",
        "Authorization" = paste("Bearer", Sys.getenv("CERVICAL_API_KEY", unset = ""))
      ),
      body   = jsonlite::toJSON(build_api_payload(patient_row), auto_unbox = TRUE),
      encode = "raw",
      httr::timeout(as.numeric(Sys.getenv("CERVICAL_API_TIMEOUT", unset = "10")))
    )
    
    if (httr::status_code(resp) == 200) {
      api_body <- httr::content(resp, as = "parsed", simplifyVector = TRUE)
      normalize_api_response(api_body, local_score)
    } else {
      local_score$api_status <- paste0("API returned status ", httr::status_code(resp), "; using local model.")
      local_score
    }
  }, error = function(e) {
    local_score$api_status <- paste0("API call failed: ", conditionMessage(e), "; using local model.")
    local_score
  })
  
  result
}

load_patients_from_db <- function() {
  dbGetQuery(con, "
    SELECT
      p.patient_id::text                      AS patient_id,
      p.first_name,
      p.last_name,
      p.first_name || ' ' || p.last_name      AS patient_name,
      p.dob::text                             AS dob,
      p.phone_contact                         AS phone,
      p.address,
      p.email,

      -- risk assessment inputs
      ra.age_at_assessment                    AS age,
      ra.first_sexual_intercourse_age         AS first_sexual_intercourse_age, 
      ra.num_sexual_partners                  AS number_of_sexual_partners,
      ra.num_pregnancies                      AS num_of_pregnancies,
      ra.smokes::int                          AS smokes,
      ra.smokes_years,
      ra.smokes_packs_year                    AS smokes_packs_per_year,
      ra.hormonal_contraceptives::int         AS hormonal_contraceptives,
      ra.hc_years                             AS hormonal_contraceptives_years,
      ra.iud::int                             AS iud,
      ra.iud_years,
      ra.std::int                             AS stds,
      ra.std_count                            AS stds_number,
      ra.dx_cancer::int                       AS dx_cancer,
      ra.dx_cin::int                          AS dx_cin,
      ra.dx_hpv::int                          AS dx_hpv,
      CASE WHEN ra.dx_cancer OR ra.dx_cin
                OR ra.dx_hpv THEN 1 ELSE 0
      END                                     AS dx,

      -- dss outputs
      rec.predicted_probability,
      initcap(rec.risk_level::text)              AS risk_level,
      initcap(rec.recommendation_category::text) AS model_recommendation,
      initcap(rec.recommendation_category::text) AS final_recommendation,

      -- placeholders for app-managed columns
      NULL::text                              AS override_reason,
      NULL::text                              AS last_reviewed,
      NULL::text                              AS top_drivers,
      NULL::text                              AS city,
      'PA'                                    AS state,
      NULL::text                              AS zip

    FROM patient p
    LEFT JOIN encounter e
      ON e.patient_id = p.patient_id
    LEFT JOIN risk_assessment ra
      ON ra.encounter_id = e.encounter_id
    LEFT JOIN dss_recommendation rec
      ON rec.assessment_id = ra.assessment_id
    WHERE p.is_active = TRUE
    ORDER BY p.last_name, p.first_name
  ")
}

# ---------------------------
# Demo patient registry derived from the dataset (demo mode only)
# ---------------------------
# Synthetic identities (no names, phones or addresses from any real person)
# attached to risk-factor rows sampled from the public UCI table.
DEMO_N_PATIENTS <- 60

build_demo_patients <- function(source_df, n = DEMO_N_PATIENTS) {
  set.seed(2026)
  idx <- sort(sample(seq_len(nrow(source_df)), min(n, nrow(source_df))))
  src <- source_df[idx, , drop = FALSE]
  n <- nrow(src)
  ids <- sprintf("DEMO-%03d", seq_len(n))
  age <- round(coalesce_num(src$age, 30))

  data.frame(
    patient_id                    = ids,
    first_name                    = "Demo",
    last_name                     = sprintf("Patient %03d", seq_len(n)),
    patient_name                  = sprintf("Demo Patient %03d", seq_len(n)),
    dob                           = as.character(as.Date(sprintf("%d-01-01", as.integer(format(Sys.Date(), "%Y")) - age))),
    phone                         = "",
    address                       = "Synthetic demo record",
    email                         = NA_character_,
    age                           = age,
    first_sexual_intercourse_age  = src$first_sexual_intercourse_age,
    number_of_sexual_partners     = src$number_of_sexual_partners,
    num_of_pregnancies            = src$num_of_pregnancies,
    smokes                        = src$smokes,
    smokes_years                  = src$smokes_years,
    smokes_packs_per_year         = src$smokes_packs_per_year,
    hormonal_contraceptives       = src$hormonal_contraceptives,
    hormonal_contraceptives_years = src$hormonal_contraceptives_years,
    iud                           = src$iud,
    iud_years                     = src$iud_years,
    stds                          = src$stds,
    stds_number                   = src$stds_number,
    dx_cancer                     = src$dx_cancer,
    dx_cin                        = src$dx_cin,
    dx_hpv                        = src$dx_hpv,
    dx                            = src$dx,
    predicted_probability         = NA_real_,
    risk_level                    = NA_character_,
    model_recommendation          = NA_character_,
    final_recommendation          = NA_character_,
    override_reason               = NA_character_,
    last_reviewed                 = NA_character_,
    top_drivers                   = NA_character_,
    city                          = NA_character_,
    state                         = NA_character_,
    zip                           = NA_character_,
    stringsAsFactors = FALSE
  )
}

if (DEMO_MODE) {
  con <- NULL
  patients_df <- build_demo_patients(clean_df)
  message("CxRisk frontend: DEMO MODE (no database). Scoring API: ", api_url())
} else {
  # Connection pool: stays open for the lifetime of the app
  con <- dbPool(
    drv      = RPostgres::Postgres(),
    host     = Sys.getenv("SUPABASE_HOST"),
    dbname   = Sys.getenv("SUPABASE_DBNAME", unset = "postgres"),
    user     = Sys.getenv("SUPABASE_USER"),
    password = Sys.getenv("SUPABASE_PASSWORD"),
    port     = as.integer(Sys.getenv("SUPABASE_PORT", unset = "6543")),
    sslmode  = Sys.getenv("SUPABASE_SSLMODE", unset = "require") # Supabase requires SSL
  )

  # Clean up the pool when the app stops
  onStop(function() poolClose(con))

  patients_df <- load_patients_from_db()
}

# top_drivers computed locally since it's not stored in the DB yet
baseline_scores <- lapply(
  seq_len(nrow(patients_df)),
  function(i) score_patient(patients_df[i, , drop = FALSE])
)

# vapply keeps the types right when the table is empty (fresh database)
patients_df$top_drivers <- vapply(
  baseline_scores,
  function(x) paste(x$top_drivers, collapse = ", "),
  character(1)
)

# if predicted_probability or risk_level missing from DB rows, fill from model
patients_df$predicted_probability <- ifelse(
  is.na(patients_df$predicted_probability),
  vapply(baseline_scores, function(x) x$probability, numeric(1)),
  patients_df$predicted_probability
)

patients_df$risk_level <- ifelse(
  is.na(patients_df$risk_level) | patients_df$risk_level == "",
  vapply(baseline_scores, function(x) x$risk_level, character(1)),
  patients_df$risk_level
)

patients_df$model_recommendation <- ifelse(
  is.na(patients_df$model_recommendation) | patients_df$model_recommendation == "",
  vapply(baseline_scores, function(x) x$recommendation, character(1)),
  patients_df$model_recommendation
)

patients_df$final_recommendation <- patients_df$model_recommendation
patients_df$override_reason      <- rep(NA_character_, nrow(patients_df))
patients_df$last_reviewed        <- rep(NA_character_, nrow(patients_df))

check_db_credentials <- function(user, password) {
  
  result <- dbGetQuery(con, "
    SELECT user_id, email, password_hash, role::text, is_active
    FROM app_user
    WHERE email = $1
      AND is_active = TRUE
    LIMIT 1
  ", params = list(user))
  
  # no matching user found
  if (nrow(result) == 0) {
    return(data.frame(result = FALSE, stringsAsFactors = FALSE))
  }
  
  # wrong password: password_hash must hold a bcrypt hash, e.g. created with
  # bcrypt::hashpw("the password") in R or crypt('the password', gen_salt('bf'))
  # with pgcrypto. Plaintext values never match.
  password_ok <- tryCatch(
    bcrypt::checkpw(password, result$password_hash[1]),
    error = function(e) FALSE
  )
  if (!isTRUE(password_ok)) {
    return(data.frame(result = FALSE, stringsAsFactors = FALSE))
  }
  
  # map database role to app role
  db_role <- result$role[1]
  app_role <- if (db_role %in% c("physician", "nurse")) "clinician" else "admin"
  
  # return data.frame with result = TRUE; shinymanager requires this format
  data.frame(
    result   = TRUE,
    role     = app_role,
    user_id  = as.character(result$user_id[1]),
    stringsAsFactors = FALSE
  )
}

# ---------------------------
# UI
# ---------------------------
ui <- navbarPage(
  title = "Cervical Cancer DSS",
  id = "main_navbar",
  header = tags$head(
    tags$style(HTML("
      body {
        background: #E5F6FF;
        color: #000000;
        font-family: 'Arial', 'Helvetica', sans-serif;
      }
      .navbar-default {
        background-color: #FFFFFF;
        border-color: #d7eaf6;
      }
      .navbar-default .navbar-brand,
      .navbar-default .navbar-nav > li > a {
        font-weight: 600;
        color: #000000 !important;
      }
      .hero-card, .panel-card, .metric-card, .mini-card {
        background: #FFFFFF;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        margin-bottom: 18px;
      }
      .hero-card {
        padding: 30px;
      }
      .hero-title {
        font-size: 30px;
        font-weight: 700;
        margin-bottom: 10px;
      }
      .hero-subtitle {
        font-size: 16px;
        line-height: 1.5;
      }
      .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 14px;
      }
      .soft-box {
        background: #E5F6FF;
        border-radius: 18px;
        padding: 16px;
      }
      .blue-btn, .btn-primary {
        background-color: #AFCBF8 !important;
        border-color: #AFCBF8 !important;
        color: #000000 !important;
        border-radius: 14px !important;
        font-weight: 700;
      }
      .red-btn, .btn-danger {
        background-color: #BC0E0E !important;
        border-color: #BC0E0E !important;
        color: #FFFFFF !important;
        border-radius: 14px !important;
        font-weight: 700;
      }
      .metric-value {
        font-size: 30px;
        font-weight: 700;
      }
      .metric-label {
        font-size: 13px;
        color: #333333;
      }
      .risk-pill {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        font-weight: 700;
        margin-right: 8px;
      }
      .risk-low { background: #d9f0e3; }
      .risk-medium { background: #fff2cc; }
      .risk-high { background: #f4cccc; }
      .driver-tag {
        display: inline-block;
        background: #E5F6FF;
        border-radius: 999px;
        padding: 7px 12px;
        margin: 4px 6px 4px 0;
        font-weight: 600;
      }
      .override-note {
        background: #fff3f3;
        border-left: 6px solid #BC0E0E;
        border-radius: 16px;
        padding: 14px 16px;
      }
      .dt-buttons, .dataTables_filter {
        margin-bottom: 8px;
      }
      .conf-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 13px;
        margin-left: 8px;
      }
      .conf-high   { background: #d9f0e3; color: #1a6637; }
      .conf-medium { background: #fff2cc; color: #7a5c00; }
      .conf-low    { background: #f4cccc; color: #7a1a1a; }
      .warning-box {
        background: #fff8e1;
        border-left: 5px solid #f59e0b;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 10px;
      }
      .warning-box p { margin: 0; font-size: 14px; }
      .warning-icon { font-size: 16px; margin-right: 6px; }
      .nl-explanation {
        background: #f0f7ff;
        border-radius: 14px;
        padding: 16px 20px;
        font-size: 15px;
        line-height: 1.6;
        margin-bottom: 16px;
      }
      .model-prob-bar-wrap {
        margin-bottom: 10px;
      }
      .model-prob-label {
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 3px;
      }
      .model-prob-track {
        background: #e5f6ff;
        border-radius: 999px;
        height: 18px;
        width: 100%;
        position: relative;
      }
      .model-prob-fill {
        height: 18px;
        border-radius: 999px;
        position: absolute;
        top: 0; left: 0;
        transition: width 0.4s;
      }
      .model-prob-val {
        position: absolute;
        right: 8px;
        top: 0;
        font-size: 12px;
        font-weight: 700;
        line-height: 18px;
      }
      .insights-section-title {
        font-size: 17px;
        font-weight: 700;
        margin: 18px 0 10px 0;
        color: #1a1a2e;
      }
    "))
  ),
  
  tabPanel(
    "Home",
    fluidPage(
      br(),
      div(
        class = "hero-card",
        fluidRow(
          column(
            8,
            div(class = "hero-title", "Cervical Cancer Risk Stratification and Screening Triage"),
            div(
              class = "hero-subtitle",
              "Reviews patient risk profiles, generate screening recommendations, and track follow-up actions across your patient panel."
            ),
            br(),
            actionButton("go_patient_search", "Open patient search", class = "blue-btn"),
            HTML("&nbsp;"),
            actionButton("go_population_dashboard", "Open population dashboard", class = "blue-btn")
          ),
          column(
            4,
            div(
              class = "soft-box",
              h4(strong("How to Use This Tool")),
              tags$ul(
                tags$li("Search for an existing patient or add a new one."),
                tags$li("Enter or confirm their risk factor information."),
                tags$li("Review the model recommendation and top risk drivers."),
                tags$li("Override if clinical judgement differs (note that a reason for override is required)."),
                tags$li("Download the patient summary for documentation.")
              )
            )
          )
        )
      ),
      fluidRow(
        column(
          4,
          div(
            class = "mini-card",
            h4(strong("Patient Triage")),
            p("Search your panel, run a risk assessment, and receive a recommendation: routine recall, expedited HPV/Pap testing, or referral for colposcopy.")
          )
        ),
        column(
          4,
          div(
            class = "mini-card",
            h4(strong("Explainable Recommendations")),
            p("Every recommendation shows the top contributing risk drivers for full transparency and review.")
          )
        ),
        column(
          4,
          div(
            class = "mini-card",
            h4(strong("Population Overview")),
            p("The population dashboard surfaces your highest-risk patients, common risk patterns, and a log of all clinician overrides.")
          )
        )
      )
    )
  ),
  
  tabPanel(
    "Patient Search",
    fluidPage(
      br(),
      fluidRow(
        column(
          8,
          div(
            class = "panel-card",
            div(class = "section-title", "Patient Search"),
            textInput("patient_search", NULL, placeholder = "Search by patient name or ID"),
            DTOutput("patient_table"),
            br(),
            actionButton("open_selected_patient", "Open selected patient", class = "blue-btn"),
            HTML("&nbsp;"),
            actionButton("toggle_add_patient", "Add new patient", class = "blue-btn")
          )
        ),
        column(
          4,
          uiOutput("selected_patient_preview")
        )
      ),
      uiOutput("add_patient_panel")
    )
  ),
  
  tabPanel(
    "Patient Risk Assessment",
    fluidPage(
      br(),
      uiOutput("assessment_panel")
    )
  ),
  
  tabPanel(
    "Model Insights",
    fluidPage(
      br(),
      uiOutput("insights_panel")
    )
  ),

  tabPanel(
    "Population Dashboard",
    fluidPage(
      br(),
      fluidRow(
        column(3, div(class = "metric-card", div(class = "metric-value", textOutput("metric_n_patients")), div(class = "metric-label", "Patients in prototype registry"))),
        column(3, div(class = "metric-card", div(class = "metric-value", textOutput("metric_abnormal")), div(class = "metric-label", "Any abnormal outcome in source data"))),
        column(3, div(class = "metric-card", div(class = "metric-value", textOutput("metric_high_risk")), div(class = "metric-label", "Current high-risk patients"))),
        column(3, div(class = "metric-card", div(class = "metric-value", textOutput("metric_overrides")), div(class = "metric-label", "Clinician overrides logged")))
      ),
      fluidRow(
        column(6, div(class = "panel-card", div(class = "section-title", "Risk level mix"), plotOutput("plot_risk_mix", height = 300))),
        column(6, div(class = "panel-card", div(class = "section-title", "Common patterns in high-risk group"), plotOutput("plot_top_drivers", height = 300)))
      ),
      fluidRow(
        column(6, div(class = "panel-card", div(class = "section-title", "Age vs. pregnancies"), plotOutput("plot_age_preg", height = 320))),
        column(6, div(class = "panel-card", div(class = "section-title", "Override activity"), DTOutput("override_table")))
      )
    )
  )
)

login_css <- tags$style(HTML("
  .panel-auth {
    background: #E5F6FF !important;
    font-family: 'Arial', 'Helvetica', sans-serif !important;
    min-height: 100vh;
  }
  .panel-auth .panel.panel-primary {
    background: #FFFFFF !important;
    border-radius: 20px !important;
    border: none !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08) !important;
  }
  .panel-auth .panel-primary {
    border-top-color: #FFFFFF !important;
  }
  .panel-auth .panel-body {
    padding: 30px !important;
    font-family: 'Arial', 'Helvetica', sans-serif !important;
  }
  .panel-auth #auth-shinymanager-auth-head {
    font-family: 'Arial', 'Helvetica', sans-serif !important;
    font-weight: 600 !important;
    color: #333333 !important;
    font-size: 16px !important;
  }
  .panel-auth .control-label {
    font-family: 'Arial', 'Helvetica', sans-serif !important;
    font-weight: 600 !important;
    color: #000000 !important;
  }
  .panel-auth .form-control {
    border-radius: 12px !important;
    border: 1px solid #d7eaf6 !important;
    font-family: 'Arial', 'Helvetica', sans-serif !important;
    padding: 8px 12px !important;
  }
  .panel-auth .form-control:focus {
    border-color: #AFCBF8 !important;
    box-shadow: 0 0 0 3px rgba(175,203,248,0.3) !important;
    outline: none !important;
  }
  .panel-auth #auth-go_auth {
    background-color: #AFCBF8 !important;
    border-color: #AFCBF8 !important;
    color: #000000 !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-family: 'Arial', 'Helvetica', sans-serif !important;
    width: 100% !important;
    padding: 10px !important;
  }
  .panel-auth #auth-go_auth:hover {
    background-color: #93b8f5 !important;
    border-color: #93b8f5 !important;
  }
"))

if (!DEMO_MODE) {
  ui <- secure_app(
    ui,
    title = "Cervical Cancer DSS: Sign In",
    tags_top = tags$div(
      style = "text-align: center; margin-bottom: 10px;",
      tags$h3(style = "font-weight: 700; color: #000;", "Cervical Cancer DSS")
    ),
    tags_bottom = tags$head(login_css)
  )
}

# ---------------------------
# Server
# ---------------------------
server <- function(input, output, session) {
  if (DEMO_MODE) {
    res_auth <- list(user = "demo", role = "clinician")
  } else {
    res_auth <- secure_server(check_credentials = check_db_credentials)
  }
  user_role <- reactive({ res_auth$role })
  
  rv <- reactiveValues(
    patients = patients_df,
    selected_patient_id = NULL,
    current_user_id = NULL,
    show_add_patient = FALSE,
    show_override = FALSE,
    api_insights = NULL,
    override_log = data.frame(
      timestamp = character(),
      patient_id = character(),
      patient_name = character(),
      model_recommendation = character(),
      final_recommendation = character(),
      reason = character(),
      stringsAsFactors = FALSE
    )
  )
  
  filtered_patients <- reactive({
    dat <- rv$patients
    q <- trimws(input$patient_search %||% "")
    if (nchar(q) == 0) return(dat)
    
    q_low <- tolower(q)
    keep <- grepl(q_low, tolower(dat$patient_name)) | grepl(q_low, tolower(dat$patient_id))
    dat[keep, , drop = FALSE]
  })
  
  output$patient_table <- renderDT({
    dat <- filtered_patients() %>%
      transmute(
        patient_name,
        age = round(coalesce_num(age, 0), 0),
        smokes = binary_label(smokes),
        stds = binary_label(stds),
        risk_level,
        recommendation = final_recommendation
      )
    
    datatable(
      dat,
      rownames = FALSE,
      selection = "single",
      options = list(pageLength = 8, autoWidth = TRUE, dom = "tip")
    )
  })
  
  observeEvent(input$go_patient_search, {
    updateNavbarPage(session, "main_navbar", selected = "Patient Search")
  })
  
  observeEvent(input$go_population_dashboard, {
    updateNavbarPage(session, "main_navbar", selected = "Population Dashboard")
  })
  
  observeEvent(input$open_selected_patient, {
    idx <- input$patient_table_rows_selected
    if (length(idx) == 1) {
      chosen <- filtered_patients()[idx, , drop = FALSE]
      rv$selected_patient_id <- chosen$patient_id[1]
      updateNavbarPage(session, "main_navbar", selected = "Patient Risk Assessment")
    }
  })
  
  observeEvent(input$toggle_add_patient, {
    rv$show_add_patient <- !isTRUE(rv$show_add_patient)
  })
  
  current_patient <- reactive({
    req(rv$selected_patient_id)
    rv$patients %>% filter(patient_id == rv$selected_patient_id)
  })
  
  output$selected_patient_preview <- renderUI({
    idx <- input$patient_table_rows_selected
    if (length(idx) != 1) {
      return(
        div(
          class = "panel-card",
          div(class = "section-title", "Selected Patient"),
          p("Select a row in the table to preview a patient.")
        )
      )
    }
    
    pdat <- filtered_patients()[idx, , drop = FALSE]
    div(
      class = "panel-card",
      div(class = "section-title", "Selected Patient"),
      p(strong(pdat$patient_name)),
      p(paste("DOB:", pdat$dob)),
      p(paste("Phone:", format_phone(pdat$phone))),
      p(paste("Address:", pdat$address)), 
      p(paste("Current risk level:", pdat$risk_level)),
      p(paste("Recommendation:", pdat$final_recommendation))
    )
  })
  
  output$add_patient_panel <- renderUI({
    if (!isTRUE(rv$show_add_patient)) return(NULL)
    
    div(
      class = "panel-card",
      div(class = "section-title", "Add New Patient"),
      fluidRow(
        column(4, textInput("new_name", "Full name", value = "")),
        column(4, dateInput("new_dob", "Date of birth", value = Sys.Date() - 35 * 365)),
        column(4, textInput("new_phone", "Phone", value = "(412) 555-0000"))
      ),
      fluidRow(
        column(6, textInput("new_address", "Address", value = "100 Clinic Way")),
        column(3, textInput("new_city", "City", value = "Pittsburgh")),
        column(3, textInput("new_zip", "ZIP", value = "15213"))
      ),
      fluidRow(
        column(3, numericInput("new_age", "Age", value = 35, min = 18, max = 100)),
        column(3, numericInput("new_partners", "Sexual partners", value = 1, min = 0)),
        column(3, numericInput("new_firstsex", "Age at first intercourse", value = 18, min = 0, max = 100)),
        column(3, numericInput("new_preg", "Pregnancies", value = 0, min = 0))
      ),
      fluidRow(
        column(3, selectInput("new_smokes", "Smokes", choices = c("No" = 0, "Yes" = 1), selected = 0)),
        column(3, numericInput("new_smokes_years", "Smoking years", value = 0, min = 0)),
        column(3, numericInput("new_smokes_packs", "Smoking packs/year", value = 0, min = 0, step = 0.1)),
        column(3, selectInput("new_hc", "Hormonal contraceptives", choices = c("No" = 0, "Yes" = 1), selected = 0))
      ),
      fluidRow(
        column(3, numericInput("new_hc_years", "Hormonal contraceptive years", value = 0, min = 0, step = 0.1)),
        column(3, selectInput("new_iud", "IUD", choices = c("No" = 0, "Yes" = 1), selected = 0)),
        column(3, numericInput("new_iud_years", "IUD years", value = 0, min = 0, step = 0.1)),
        column(3, selectInput("new_stds", "Any STDs", choices = c("No" = 0, "Yes" = 1), selected = 0))
      ),
      fluidRow(
        column(4, numericInput("new_stds_number", "Number of STDs", value = 0, min = 0)),
        column(4, selectInput("new_dx_hpv", "Prior HPV diagnosis", choices = c("No" = 0, "Yes" = 1), selected = 0)),
        column(4, selectInput("new_dx_cancer", "Prior cancer diagnosis", choices = c("No" = 0, "Yes" = 1), selected = 0))
      ),
      actionButton("save_new_patient", "Save patient", class = "blue-btn")
    )
  })
  
  observeEvent(input$save_new_patient, {
    tryCatch({
    parts <- strsplit(trimws(input$new_name), "\\s+")[[1]]
    first_name <- if (length(parts) >= 1 && nzchar(parts[1])) parts[1] else "New"
    last_name  <- if (length(parts) >= 2) paste(parts[-1], collapse = " ") else "Patient"

    if (DEMO_MODE) {
    # Demo mode: score the new patient and keep it in memory for this session
    new_patient_id <- sprintf("DEMO-NEW-%03d", sum(grepl("^DEMO-NEW-", rv$patients$patient_id)) + 1)
    new_row <- data.frame(
      patient_id                    = new_patient_id,
      first_name                    = first_name,
      last_name                     = last_name,
      patient_name                  = paste(first_name, last_name),
      dob                           = as.character(input$new_dob),
      phone                         = input$new_phone,
      address                       = input$new_address,
      email                         = NA_character_,
      age                           = input$new_age,
      number_of_sexual_partners     = input$new_partners,
      first_sexual_intercourse_age  = input$new_firstsex,
      num_of_pregnancies            = input$new_preg,
      smokes                        = as.numeric(input$new_smokes),
      smokes_years                  = input$new_smokes_years,
      smokes_packs_per_year         = input$new_smokes_packs,
      hormonal_contraceptives       = as.numeric(input$new_hc),
      hormonal_contraceptives_years = input$new_hc_years,
      iud                           = as.numeric(input$new_iud),
      iud_years                     = input$new_iud_years,
      stds                          = as.numeric(input$new_stds),
      stds_number                   = input$new_stds_number,
      dx_cancer                     = as.numeric(input$new_dx_cancer),
      dx_cin                        = 0,
      dx_hpv                        = as.numeric(input$new_dx_hpv),
      dx                            = as.numeric(as.numeric(input$new_dx_hpv) + as.numeric(input$new_dx_cancer) > 0),
      city                          = input$new_city,
      state                         = NA_character_,
      zip                           = input$new_zip,
      stringsAsFactors = FALSE
    )
    sc <- call_risk_api(new_row)
    new_row$predicted_probability <- sc$probability
    new_row$risk_level            <- sc$risk_level
    new_row$model_recommendation  <- sc$recommendation
    new_row$final_recommendation  <- sc$recommendation
    new_row$override_reason       <- NA_character_
    new_row$last_reviewed         <- as.character(Sys.time())
    new_row$top_drivers           <- paste(sc$top_drivers, collapse = ", ")

    rv$patients <- dplyr::bind_rows(rv$patients, new_row)
    rv$api_insights <- sc
    rv$selected_patient_id <- new_patient_id
    updateNavbarPage(session, "main_navbar", selected = "Patient Risk Assessment")
    } else {
    # 1. insert into patient, get back the generated UUID
    new_patient <- dbGetQuery(con, "
    INSERT INTO patient (patient_id, first_name, last_name, dob, phone_contact, address, is_active)
    VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, TRUE)
    RETURNING patient_id::text
    ", params = list(
      first_name,
      last_name,
      as.character(input$new_dob),
      input$new_phone,
      input$new_address
    ))
    new_patient_id <- new_patient$patient_id[1]
    
    req(res_auth$user)
    
    current_user <- dbGetQuery(con, "
    SELECT user_id::text FROM app_user 
    WHERE email = $1 
    LIMIT 1
  ", params = list(res_auth$user))
    
    if (nrow(current_user) == 0) stop(paste("No app_user found for email:", res_auth$user))
    current_user_id <- current_user$user_id[1]
    
    # 2. insert into encounter using the new patient_id
    new_encounter <- dbGetQuery(con, "
    INSERT INTO encounter (encounter_id, patient_id, clinician_id, encounter_date)
    VALUES (gen_random_uuid(), $1::uuid, $2::uuid, CURRENT_DATE)
    RETURNING encounter_id::text
  ", params = list(new_patient_id, current_user_id))
    new_encounter_id <- new_encounter$encounter_id[1]
    
    # 3. insert into risk_assessment using the new encounter_id
    new_assessment <- dbGetQuery(con, "
    INSERT INTO risk_assessment (
      assessment_id, encounter_id, age_at_assessment, num_sexual_partners,
      num_pregnancies, smokes, smokes_years, smokes_packs_year,
      hormonal_contraceptives, hc_years, iud, iud_years,
      std, std_count, dx_cancer, dx_cin, dx_hpv,
      first_sexual_intercourse_age
    )
    VALUES (
      gen_random_uuid(), $1::uuid, $2, $3, $4, $5, $6, $7,
      $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
    )
    RETURNING assessment_id::text
  ", params = list(
    new_encounter_id,
    input$new_age,
    input$new_partners,
    input$new_preg,
    as.logical(as.numeric(input$new_smokes)),
    input$new_smokes_years,
    input$new_smokes_packs,
    as.logical(as.numeric(input$new_hc)),
    input$new_hc_years,
    as.logical(as.numeric(input$new_iud)),
    input$new_iud_years,
    as.logical(as.numeric(input$new_stds)),
    input$new_stds_number,
    as.logical(as.numeric(input$new_dx_cancer)),
    FALSE,  # dx_cin: not in the add patient form
    as.logical(as.numeric(input$new_dx_hpv)),
    input$new_firstsex
  ))
    new_assessment_id <- new_assessment$assessment_id[1]
    
    # 4. score the patient locally
    new_row <- data.frame(
      age                          = input$new_age,
      number_of_sexual_partners    = input$new_partners,
      first_sexual_intercourse_age = input$new_firstsex,
      num_of_pregnancies           = input$new_preg,
      smokes                       = as.numeric(input$new_smokes),
      smokes_years                 = input$new_smokes_years,
      smokes_packs_per_year        = input$new_smokes_packs,
      hormonal_contraceptives      = as.numeric(input$new_hc),
      hormonal_contraceptives_years = input$new_hc_years,
      iud                          = as.numeric(input$new_iud),
      iud_years                    = input$new_iud_years,
      stds                         = as.numeric(input$new_stds),
      stds_number                  = input$new_stds_number,
      dx_cancer                    = as.numeric(input$new_dx_cancer),
      dx_cin                       = 0,
      dx_hpv                       = as.numeric(input$new_dx_hpv),
      dx                           = as.numeric(input$new_dx_hpv) + as.numeric(input$new_dx_cancer) > 0
    )
    sc <- call_risk_api(new_row)
    
    # 5. insert into dss_recommendation using the new assessment_id
    dbExecute(con, "
    INSERT INTO dss_recommendation (
      assessment_id, predicted_probability,
      risk_level, recommendation_category, model_version
    )
    VALUES ($1::uuid, $2, $3, $4, $5)
  ", params = list(
    new_assessment_id,
    sc$probability,
    tolower(sc$risk_level),   # store as lowercase to match existing rows
    sc$recommendation,
    "logistic_v1"
  ))
    
    # 6. query just the new patient and append
    new_display_row <- tryCatch({
      dbGetQuery(con, "
        SELECT
          p.patient_id::text                         AS patient_id,
          p.first_name,
          p.last_name,
          p.first_name || ' ' || p.last_name         AS patient_name,
          p.dob::text                                AS dob,
          p.phone_contact                            AS phone,
          p.address,
          p.email,
          ra.age_at_assessment                       AS age,
          ra.first_sexual_intercourse_age,
          ra.num_sexual_partners                     AS number_of_sexual_partners,
          ra.num_pregnancies                         AS num_of_pregnancies,
          ra.smokes::int                             AS smokes,
          ra.smokes_years,
          ra.smokes_packs_year                       AS smokes_packs_per_year,
          ra.hormonal_contraceptives::int            AS hormonal_contraceptives,
          ra.hc_years                                AS hormonal_contraceptives_years,
          ra.iud::int                                AS iud,
          ra.iud_years,
          ra.std::int                                AS stds,
          ra.std_count                               AS stds_number,
          ra.dx_cancer::int                          AS dx_cancer,
          ra.dx_cin::int                             AS dx_cin,
          ra.dx_hpv::int                             AS dx_hpv,
          CASE WHEN ra.dx_cancer OR ra.dx_cin
                    OR ra.dx_hpv THEN 1 ELSE 0 END  AS dx,
          rec.predicted_probability,
          initcap(rec.risk_level::text)              AS risk_level,
          initcap(rec.recommendation_category::text) AS model_recommendation,
          initcap(rec.recommendation_category::text) AS final_recommendation,
          NULL::text                                 AS override_reason,
          NULL::text                                 AS last_reviewed,
          NULL::text                                 AS top_drivers,
          NULL::text                                 AS city,
          'PA'                                       AS state,
          NULL::text                                 AS zip
        FROM patient p
        LEFT JOIN encounter e ON e.patient_id = p.patient_id
        LEFT JOIN risk_assessment ra ON ra.encounter_id = e.encounter_id
        LEFT JOIN dss_recommendation rec ON rec.assessment_id = ra.assessment_id
        WHERE p.patient_id = $1::uuid
      ", params = list(new_patient_id))
    }, error = function(e) {
      message("Could not query new patient: ", conditionMessage(e))
      NULL
    })
    
    Sys.sleep(0.3)
    rv$patients <- load_patients_from_db()
    
    # 7. navigate to the new patient's assessment
    rv$selected_patient_id <- new_patient_id
    rv$show_add_patient    <- TRUE
    updateNavbarPage(session, "main_navbar", selected = "Patient Risk Assessment")
    }

    }, error = function(e) {
      message("ERROR in save_new_patient: ", conditionMessage(e))
      showNotification(paste("Save failed:", conditionMessage(e)), type = "error", duration = 10)
    })
  })
  
  output$assessment_panel <- renderUI({
    if (is.null(rv$selected_patient_id)) {
      return(
        div(
          class = "panel-card",
          div(class = "section-title", "Patient Risk Assessment"),
          p("Open a patient from the search screen to view and run an assessment.")
        )
      )
    }
    
    pdat <- current_patient()
    
    div(
      fluidRow(
        column(
          4,
          div(
            class = "panel-card",
            div(class = "section-title", "Patient Risk Assessment"),
            p(strong(pdat$patient_name)),
            p(paste("DOB:", pdat$dob)),
            p(paste("Address:", pdat$address, pdat$city, pdat$state, pdat$zip)),
            numericInput("assess_age", "Age", value = round(coalesce_num(pdat$age, 35), 0), min = 18, max = 100),
            numericInput("assess_partners", "Number of sexual partners", value = round(coalesce_num(pdat$number_of_sexual_partners, 1), 0), min = 0),
            numericInput("assess_firstsex", "Age at first sexual intercourse", value = round(coalesce_num(pdat$first_sexual_intercourse_age, 18), 0), min = 0, max = 100),
            numericInput("assess_preg", "Number of pregnancies", value = round(coalesce_num(pdat$num_of_pregnancies, 0), 0), min = 0),
            selectInput("assess_smokes", "Smokes", choices = c("No" = 0, "Yes" = 1), selected = as.character(round(coalesce_num(pdat$smokes, 0), 0))),
            numericInput("assess_smokes_years", "Smoking years", value = coalesce_num(pdat$smokes_years, 0), min = 0),
            numericInput("assess_smokes_packs", "Smoking packs/year", value = coalesce_num(pdat$smokes_packs_per_year, 0), min = 0, step = 0.1),
            selectInput("assess_hc", "Hormonal contraceptives", choices = c("No" = 0, "Yes" = 1), selected = as.character(round(coalesce_num(pdat$hormonal_contraceptives, 0), 0))),
            numericInput("assess_hc_years", "Hormonal contraceptive years", value = coalesce_num(pdat$hormonal_contraceptives_years, 0), min = 0, step = 0.1),
            selectInput("assess_iud", "IUD", choices = c("No" = 0, "Yes" = 1), selected = as.character(round(coalesce_num(pdat$iud, 0), 0))),
            numericInput("assess_iud_years", "IUD years", value = coalesce_num(pdat$iud_years, 0), min = 0, step = 0.1),
            selectInput("assess_stds", "Any STDs", choices = c("No" = 0, "Yes" = 1), selected = as.character(round(coalesce_num(pdat$stds, 0), 0))),
            numericInput("assess_stds_number", "Number of STDs", value = round(coalesce_num(pdat$stds_number, 0), 0), min = 0),
            selectInput("assess_dx_hpv", "Prior HPV diagnosis", choices = c("No" = 0, "Yes" = 1), selected = as.character(round(coalesce_num(pdat$dx_hpv, 0), 0))),
            selectInput("assess_dx_cancer", "Prior cancer diagnosis", choices = c("No" = 0, "Yes" = 1), selected = as.character(round(coalesce_num(pdat$dx_cancer, 0), 0))),
            sliderInput("low_threshold", "Routine / expedited threshold", min = 0.05, max = 0.40, value = 0.15, step = 0.01),
            sliderInput("high_threshold", "Expedited / referral threshold", min = 0.20, max = 0.80, value = 0.35, step = 0.01),
            actionButton("run_assessment", "Run risk assessment", class = "blue-btn")
          )
        ),
        column(
          8,
          div(
            class = "panel-card",
            div(class = "section-title", "Assessment Output"),
            uiOutput("risk_summary_ui"),
            plotOutput("risk_bar_plot", height = 160),
            h4(strong("Top risk drivers")),
            uiOutput("driver_tags_ui"),
            br(),
            h4(strong("Recommended next step")),
            uiOutput("recommendation_ui"),
            br(),
            actionButton("toggle_override_panel", "Clinician override", class = "red-btn"),
            HTML("&nbsp;"),
            downloadButton("download_patient_summary", "Download patient summary")
          ),
          uiOutput("override_panel")
        )
      )
    )
  })
  
  observeEvent(input$toggle_override_panel, {
    rv$show_override <- !isTRUE(rv$show_override)
  })
  
  observeEvent(input$run_assessment, {
    req(rv$selected_patient_id)
    idx <- which(rv$patients$patient_id == rv$selected_patient_id)
    req(length(idx) == 1)
    
    rv$patients$age[idx] <- input$assess_age
    rv$patients$number_of_sexual_partners[idx] <- input$assess_partners
    rv$patients$first_sexual_intercourse_age[idx] <- input$assess_firstsex
    rv$patients$num_of_pregnancies[idx] <- input$assess_preg
    rv$patients$smokes[idx] <- as.numeric(input$assess_smokes)
    rv$patients$smokes_years[idx] <- input$assess_smokes_years
    rv$patients$smokes_packs_per_year[idx] <- input$assess_smokes_packs
    rv$patients$hormonal_contraceptives[idx] <- as.numeric(input$assess_hc)
    rv$patients$hormonal_contraceptives_years[idx] <- input$assess_hc_years
    rv$patients$iud[idx] <- as.numeric(input$assess_iud)
    rv$patients$iud_years[idx] <- input$assess_iud_years
    rv$patients$stds[idx] <- as.numeric(input$assess_stds)
    rv$patients$stds_number[idx] <- input$assess_stds_number
    rv$patients$dx_hpv[idx] <- as.numeric(input$assess_dx_hpv)
    rv$patients$dx_cancer[idx] <- as.numeric(input$assess_dx_cancer)
    rv$patients$dx[idx] <- ifelse(as.numeric(input$assess_dx_hpv) + as.numeric(input$assess_dx_cancer) > 0, 1, 0)
    
    sc <- call_risk_api(rv$patients[idx, , drop = FALSE], input$low_threshold, input$high_threshold)
    rv$patients$predicted_probability[idx] <- sc$probability
    rv$patients$risk_level[idx] <- sc$risk_level
    rv$patients$model_recommendation[idx] <- sc$recommendation
    # Only replace final recommendation if not overridden yet
    if (is.na(rv$patients$override_reason[idx]) || !nzchar(rv$patients$override_reason[idx])) {
      rv$patients$final_recommendation[idx] <- sc$recommendation
    }
    rv$patients$top_drivers[idx] <- paste(sc$top_drivers, collapse = ", ")
    rv$patients$last_reviewed[idx] <- as.character(Sys.time())
    
    # Store full API response for Model Insights tab
    rv$api_insights <- sc
  })
  
  output$risk_summary_ui <- renderUI({
    req(rv$selected_patient_id)
    pdat <- current_patient()
    risk_level <- pdat$risk_level[1]
    pill_class <- if (risk_level == "Low") "risk-pill risk-low" else if (risk_level == "Medium") "risk-pill risk-medium" else "risk-pill risk-high"
    
    # Confidence badge from API insights if available
    conf_badge <- NULL
    if (!is.null(rv$api_insights) && !is.null(rv$api_insights$confidence_tier)) {
      tier <- rv$api_insights$confidence_tier
      badge_class <- if (grepl("High", tier, ignore.case = TRUE)) "conf-badge conf-high" else
                     if (grepl("Low",  tier, ignore.case = TRUE)) "conf-badge conf-low"  else
                     "conf-badge conf-medium"
      conf_val <- if (!is.null(rv$api_insights$confidence)) paste0(" (", round(rv$api_insights$confidence * 100, 0), "%)") else ""
      conf_badge <- tags$span(class = badge_class, paste0(tier, " Confidence", conf_val))
    }
    
    # Warnings from API if available
    warnings_ui <- NULL
    if (!is.null(rv$api_insights) && length(rv$api_insights$warnings) > 0) {
      w <- rv$api_insights$warnings
      w <- w[nzchar(trimws(w))]
      if (length(w) > 0) {
        warnings_ui <- tagList(
          tags$div(style = "margin-top: 14px;", tags$strong("⚠ Clinical Alerts")),
          tagList(lapply(w, function(msg) {
            tags$div(class = "warning-box", tags$p(tags$span(class = "warning-icon", "⚠"), msg))
          }))
        )
      }
    }
    
    div(
      div(
        span(class = pill_class, risk_level),
        conf_badge,
        tags$span(style = "font-size: 20px; font-weight: 700; margin-left: 10px;",
                  scales::percent(coalesce_num(pdat$predicted_probability[1], 0), accuracy = 0.1))
      ),
      p(style = "margin-top: 8px;", paste("Model recommendation:", pdat$model_recommendation[1])),
      p(paste("Final recommendation:", pdat$final_recommendation[1])),
      warnings_ui
    )
  })
  
  output$driver_tags_ui <- renderUI({
    req(rv$selected_patient_id)
    pdat <- current_patient()
    drivers <- unlist(strsplit(pdat$top_drivers[1], ",\\s*"))
    tagList(lapply(drivers, function(x) tags$span(class = "driver-tag", x)))
  })
  
  output$recommendation_ui <- renderUI({
    req(rv$selected_patient_id)
    pdat <- current_patient()
    
    if (!is.na(pdat$override_reason[1]) && nzchar(pdat$override_reason[1])) {
      div(
        class = "override-note",
        p(strong("Clinician override applied")),
        p(paste("Final recommendation:", pdat$final_recommendation[1])),
        p(paste("Reason:", pdat$override_reason[1]))
      )
    } else {
      div(
        class = "soft-box",
        p(strong(pdat$final_recommendation[1])),
        p("Recommendation is generated from the current risk estimate and threshold settings.")
      )
    }
  })
  
  output$risk_bar_plot <- renderPlot({
    req(rv$selected_patient_id)
    pdat <- current_patient()
    risk <- coalesce_num(pdat$predicted_probability[1], 0)
    
    ggplot(data.frame(label = "Risk", value = risk), aes(x = label, y = value, fill = label)) +
      geom_col(width = 0.55, show.legend = FALSE) +
      coord_flip() +
      scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
      labs(x = NULL, y = "Predicted probability") +
      theme_minimal(base_size = 13) +
      theme(
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white", colour = NA),
        panel.background = element_rect(fill = "white", colour = NA)
      )
  })
  
  output$override_panel <- renderUI({
    if (!isTRUE(rv$show_override) || is.null(rv$selected_patient_id)) return(NULL)
    
    pdat <- current_patient()
    div(
      class = "panel-card",
      div(class = "section-title", "Clinician Override"),
      selectInput(
        "override_recommendation",
        "Updated recommendation",
        choices = c("Routine recall", "Expedited HPV/Pap testing", "Referral for further evaluation"),
        selected = pdat$final_recommendation[1]
      ),
      textAreaInput("override_reason_text", "Override rationale", rows = 4, placeholder = "Explain the clinical reason for overriding the model output."),
      actionButton("save_override", "Save override", class = "red-btn")
    )
  })
  
  observeEvent(input$save_override, {
    req(rv$selected_patient_id)
    idx <- which(rv$patients$patient_id == rv$selected_patient_id)
    req(length(idx) == 1)
    
    rv$patients$final_recommendation[idx] <- input$override_recommendation
    rv$patients$override_reason[idx] <- input$override_reason_text
    rv$patients$last_reviewed[idx] <- as.character(Sys.time())
    
    rv$override_log <- bind_rows(
      rv$override_log,
      data.frame(
        timestamp = as.character(Sys.time()),
        patient_id = rv$patients$patient_id[idx],
        patient_name = rv$patients$patient_name[idx],
        model_recommendation = rv$patients$model_recommendation[idx],
        final_recommendation = input$override_recommendation,
        reason = input$override_reason_text,
        stringsAsFactors = FALSE
      )
    )
  })
  
  output$download_patient_summary <- downloadHandler(
    filename = function() {
      req(rv$selected_patient_id)
      paste0("patient_summary_", rv$selected_patient_id, ".html")
    },
    content = function(file) {
      req(rv$selected_patient_id)
      pdat <- current_patient()[1, , drop = FALSE]
      
      html <- paste0(
        "<html><head><meta charset='utf-8'><title>Patient Summary</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;padding:28px;} h1,h2{margin-bottom:8px;} .box{background:#f7fbff;border-radius:16px;padding:18px;margin-bottom:16px;} .pill{display:inline-block;padding:8px 14px;border-radius:999px;background:#e5f6ff;font-weight:700;}</style>",
        "</head><body>",
        "<h1>Cervical Cancer DSS Patient Summary</h1>",
        "<div class='box'><h2>", pdat$patient_name[1], " (", pdat$patient_id[1], ")</h2>",
        "<p><strong>DOB:</strong> ", pdat$dob[1], "<br>",
        "<strong>Address:</strong> ", pdat$address[1], "<br>",
        "<strong>Phone:</strong> ", format_phone(pdat$phone[1]), "</p></div>",
        "<div class='box'><h2>Clinical Inputs</h2>",
        "<p><strong>Age:</strong> ", pdat$age[1], "<br>",
        "<strong>Smoking:</strong> ", binary_label(pdat$smokes[1]), " (years: ", round(coalesce_num(pdat$smokes_years[1], 0), 1), ", packs/year: ", round(coalesce_num(pdat$smokes_packs_per_year[1], 0), 2), ")<br>",
        "<strong>Sexual partners:</strong> ", round(coalesce_num(pdat$number_of_sexual_partners[1], 0), 0), "<br>",
        "<strong>Pregnancies:</strong> ", round(coalesce_num(pdat$num_of_pregnancies[1], 0), 0), "<br>",
        "<strong>Hormonal contraceptives:</strong> ", binary_label(pdat$hormonal_contraceptives[1]), " (years: ", round(coalesce_num(pdat$hormonal_contraceptives_years[1], 0), 1), ")<br>",
        "<strong>IUD:</strong> ", binary_label(pdat$iud[1]), " (years: ", round(coalesce_num(pdat$iud_years[1], 0), 1), ")<br>",
        "<strong>STDs:</strong> ", binary_label(pdat$stds[1]), " (number: ", round(coalesce_num(pdat$stds_number[1], 0), 0), ")<br>",
        "<strong>Prior HPV diagnosis:</strong> ", binary_label(pdat$dx_hpv[1]), "<br>",
        "<strong>Prior cancer diagnosis:</strong> ", binary_label(pdat$dx_cancer[1]), "</p></div>",
        "<div class='box'><h2>Risk Assessment</h2>",
        "<p><span class='pill'>", pdat$risk_level[1], "</span></p>",
        "<p><strong>Predicted probability:</strong> ", percent(coalesce_num(pdat$predicted_probability[1], 0), accuracy = 0.1), "<br>",
        "<strong>Top risk drivers:</strong> ", pdat$top_drivers[1], "<br>",
        "<strong>Model recommendation:</strong> ", pdat$model_recommendation[1], "<br>",
        "<strong>Final recommendation:</strong> ", pdat$final_recommendation[1], "</p>",
        if (!is.na(pdat$override_reason[1]) && nzchar(pdat$override_reason[1])) paste0("<p><strong>Override rationale:</strong> ", pdat$override_reason[1], "</p>") else "",
        "</div>",
        "<p><em>Note: this is a prototype clinician-facing summary generated for course project demonstration.</em></p>",
        "</body></html>"
      )
      writeLines(html, con = file)
    }
  )
  
  # Population dashboard outputs
  output$metric_n_patients <- renderText({
    format(nrow(rv$patients), big.mark = ",")
  })
  
  output$metric_abnormal <- renderText({
    if ("abnormal_any" %in% names(clean_df)) {
      percent(mean(coalesce_num(clean_df$abnormal_any, 0)), accuracy = 0.1)
    } else {
      "N/A"
    }
  })
  
  output$metric_high_risk <- renderText({
    sum(rv$patients$risk_level == "High", na.rm = TRUE)
  })
  
  output$metric_overrides <- renderText({
    nrow(rv$override_log)
  })
  
  output$plot_risk_mix <- renderPlot({
    plot_df <- rv$patients %>%
      count(risk_level) %>%
      mutate(risk_level = factor(risk_level, levels = c("Low", "Medium", "High")))
    
    ggplot(plot_df, aes(x = risk_level, y = n, fill = risk_level)) +
      geom_col(show.legend = FALSE) +
      labs(x = NULL, y = "Patients") +
      theme_minimal(base_size = 13) +
      theme(
        plot.background = element_rect(fill = "white", colour = NA),
        panel.background = element_rect(fill = "white", colour = NA)
      )
  })
  
  output$plot_top_drivers <- renderPlot({
    high_risk <- rv$patients %>% filter(risk_level == "High")
    if (nrow(high_risk) == 0) {
      plot.new()
      text(0.5, 0.5, "No high-risk patients in current view.")
      return()
    }
    
    summary_df <- data.frame(
      label = c("Smokes", "Any STD", "Prior HPV Dx", "Prior cancer Dx", "IUD use", "Hormonal contraceptive use"),
      value = c(
        mean(coalesce_num(high_risk$smokes, 0)),
        mean(coalesce_num(high_risk$stds, 0)),
        mean(coalesce_num(high_risk$dx_hpv, 0)),
        mean(coalesce_num(high_risk$dx_cancer, 0)),
        mean(coalesce_num(high_risk$iud, 0)),
        mean(coalesce_num(high_risk$hormonal_contraceptives, 0))
      )
    )
    
    ggplot(summary_df, aes(x = reorder(label, value), y = value, fill = label)) +
      geom_col(show.legend = FALSE) +
      coord_flip() +
      scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
      labs(x = NULL, y = "Prevalence within high-risk group") +
      theme_minimal(base_size = 13) +
      theme(
        plot.background = element_rect(fill = "white", colour = NA),
        panel.background = element_rect(fill = "white", colour = NA)
      )
  })
  
  output$plot_age_preg <- renderPlot({
    ggplot(rv$patients, aes(x = coalesce_num(age, 0), y = coalesce_num(num_of_pregnancies, 0), color = risk_level)) +
      geom_point(alpha = 0.65) +
      labs(x = "Age", y = "Number of pregnancies") +
      theme_minimal(base_size = 13) +
      theme(
        plot.background = element_rect(fill = "white", colour = NA),
        panel.background = element_rect(fill = "white", colour = NA)
      )
  })
  
  output$override_table <- renderDT({
    dat <- rv$override_log
    if (nrow(dat) == 0) {
      dat <- data.frame(
        timestamp = "No overrides recorded yet",
        patient_id = "",
        patient_name = "",
        model_recommendation = "",
        final_recommendation = "",
        reason = "",
        stringsAsFactors = FALSE
      )
    }
    datatable(dat, rownames = FALSE, options = list(pageLength = 5, autoWidth = TRUE, dom = "tip"))
  })
  # ── Model Insights tab ────────────────────────────────────────────────────────
  
  output$insights_panel <- renderUI({
    if (is.null(rv$selected_patient_id)) {
      return(div(
        class = "panel-card",
        div(class = "section-title", "Model Insights"),
        p("Run a risk assessment for a patient first to see model insights here.")
      ))
    }
    
    ins <- rv$api_insights
    
    # If no assessment has been run yet at all
    if (is.null(ins)) {
      return(div(
        class = "panel-card",
        div(class = "section-title", "Model Insights"),
        p("Run a risk assessment on the Patient Risk Assessment tab to populate this view.")
      ))
    }
    
    # If local model was used (API not configured or failed), show what we have
    api_available <- isTRUE(nzchar(ins$confidence_tier %||% ""))
    
    if (!api_available) {
      pdat <- current_patient()
      drivers <- unlist(strsplit(pdat$top_drivers[1], ",\\s*"))
      return(div(
        class = "panel-card",
        div(class = "section-title", paste("Model Insights:", pdat$patient_name[1])),
        div(
          class = "soft-box",
          style = "margin-bottom: 14px;",
          p(tags$strong("ℹ Local model used")),
          p("The Python API is not configured or could not be reached. Extended insights (confidence score, natural language explanation, population charts) are only available when the API is running."),
          p(paste("API status:", ins$api_status %||% "Local model used."))
        ),
        p(tags$strong("Risk level: "), ins$risk_level),
        p(tags$strong("Predicted probability: "), scales::percent(ins$probability, accuracy = 0.1)),
        p(tags$strong("Top drivers: "), paste(drivers, collapse = ", ")),
        p(tags$strong("Recommendation: "), ins$recommendation),
        br(),
        p(tags$em("To unlock full insights, make sure the API is running on port 8001 and CERVICAL_API_URL points to it (demo mode defaults to http://localhost:8001/predict)."))
      ))
    }
    
    pdat <- current_patient()
    
    # Confidence tier badge
    tier <- ins$confidence_tier
    badge_class <- if (grepl("High", tier, ignore.case = TRUE)) "conf-badge conf-high" else
                   if (grepl("Low",  tier, ignore.case = TRUE)) "conf-badge conf-low"  else
                   "conf-badge conf-medium"
    
    # Warnings block
    warnings_block <- NULL
    w <- ins$warnings
    if (!is.null(w)) w <- w[nzchar(trimws(w))]
    if (!is.null(w) && length(w) > 0) {
      warnings_block <- div(
        class = "panel-card",
        div(class = "insights-section-title", "⚠ Clinical Alerts"),
        tagList(lapply(w, function(msg) {
          div(class = "warning-box", p(tags$span(class = "warning-icon", "⚠"), msg))
        }))
      )
    }
    
    # NL explanation block
    nl_block <- NULL
    if (!is.null(ins$nl_explanation) && nzchar(trimws(ins$nl_explanation))) {
      nl_block <- div(
        class = "panel-card",
        div(class = "insights-section-title", "Model Explanation"),
        div(class = "nl-explanation", ins$nl_explanation)
      )
    }
    
    # Per-model probabilities block
    model_probs_block <- NULL
    if (!is.null(ins$model_probs) && length(ins$model_probs) > 0) {
      mp <- ins$model_probs
      bars <- lapply(names(mp), function(nm) {
        val <- as.numeric(mp[[nm]])
        pct <- paste0(round(val * 100, 1), "%")
        fill_col <- if (val >= 0.35) "#f4a3a3" else if (val >= 0.15) "#ffe08a" else "#9dd6b5"
        div(
          class = "model-prob-bar-wrap",
          div(class = "model-prob-label", nm),
          div(
            class = "model-prob-track",
            div(class = "model-prob-fill",
                style = paste0("width:", round(val * 100, 1), "%; background:", fill_col, ";")),
            div(class = "model-prob-val", pct)
          )
        )
      })
      model_probs_block <- div(
        class = "panel-card",
        div(class = "insights-section-title", "Per-Model Probabilities"),
        p(style = "font-size: 13px; color: #555; margin-bottom: 12px;",
          "Each bar shows the individual model's predicted probability. Disagreement between models is a signal worth noting."),
        tagList(bars)
      )
    }
    
    # Charts block: only rendered if the API returned base64 images
    charts_block <- NULL
    has_scatter      <- !is.null(ins$scatter_png_b64)      && nzchar(ins$scatter_png_b64)
    has_waterfall    <- !is.null(ins$waterfall_png_b64)    && nzchar(ins$waterfall_png_b64)
    has_contribution <- !is.null(ins$contribution_png_b64) && nzchar(ins$contribution_png_b64)
    
    if (has_scatter || has_waterfall || has_contribution) {
      chart_items <- list()
      if (has_scatter) {
        chart_items <- c(chart_items, list(
          div(class = "panel-card",
            div(class = "insights-section-title", "Population Risk Map"),
            p(style = "font-size: 13px; color: #555; margin-bottom: 10px;",
              "Where this patient sits relative to all patients in the dataset. The gold star marks the current patient."),
            tags$img(src = paste0("data:image/png;base64,", ins$scatter_png_b64),
                     style = "width:100%; border-radius: 12px;")
          )
        ))
      }
      if (has_waterfall) {
        chart_items <- c(chart_items, list(
          div(class = "panel-card",
            div(class = "insights-section-title", "Confidence Waterfall"),
            p(style = "font-size: 13px; color: #555; margin-bottom: 10px;",
              "Shows what is driving the model's confidence (or uncertainty) in this prediction."),
            tags$img(src = paste0("data:image/png;base64,", ins$waterfall_png_b64),
                     style = "width:100%; border-radius: 12px;")
          )
        ))
      }
      if (has_contribution) {
        chart_items <- c(chart_items, list(
          div(class = "panel-card",
            div(class = "insights-section-title", "Model Contribution Waterfall"),
            p(style = "font-size: 13px; color: #555; margin-bottom: 10px;",
              "How much each model contributed to the final ensemble prediction. Wide disagreement between models warrants extra caution."),
            tags$img(src = paste0("data:image/png;base64,", ins$contribution_png_b64),
                     style = "width:100%; border-radius: 12px;")
          )
        ))
      }
      charts_block <- tagList(chart_items)
    }
    
    # Header summary row
    header <- div(
      class = "panel-card",
      fluidRow(
        column(6,
          div(class = "section-title", paste("Model Insights:", pdat$patient_name[1])),
          p(paste("Assessment run:", pdat$last_reviewed[1]))
        ),
        column(6,
          div(style = "text-align: right; padding-top: 10px;",
            tags$span(style = "font-size: 15px; font-weight: 600;", "Confidence: "),
            tags$span(class = badge_class, paste0(tier, " (", round(ins$confidence * 100, 0), "%)"))
          )
        )
      )
    )
    
    tagList(header, warnings_block, nl_block, model_probs_block, charts_block)
  })

}

shinyApp(ui = ui, server = server)