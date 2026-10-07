# Install the R packages used by frontend/app.R.
# Usage (from the repository root):  Rscript frontend/install_packages.R
#        add --demo-only to skip the database packages.

repos <- "https://cloud.r-project.org"

# Needed in both demo mode and DB mode
core_pkgs <- c(
  "shiny",
  "DT",
  "dplyr",
  "ggplot2",
  "scales",
  "httr",
  "jsonlite"
)

# Needed only in DB mode (SUPABASE_HOST set): login, Postgres pool, bcrypt
db_pkgs <- c(
  "shinymanager",
  "DBI",
  "RPostgres",
  "pool",
  "bcrypt"
)

args <- commandArgs(trailingOnly = TRUE)
pkgs <- if ("--demo-only" %in% args) core_pkgs else c(core_pkgs, db_pkgs)

missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  install.packages(missing, repos = repos)
} else {
  message("All packages already installed.")
}
