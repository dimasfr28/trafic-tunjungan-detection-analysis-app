# ==============================================================================
# FUZZY CLUSTERING UNTUK VEHICLE COUNTS DATA
# FCM, PCM, FPCM, dan MFPCM Analysis
# ==============================================================================

# Load required libraries
library(factoextra)
library(dplyr)
library(cluster)
library(ppclust)
library(fclust)
library(psych)
library(clusterSim)

# ==============================================================================
# 1. LOAD DAN PREPROCESSING DATA
# ==============================================================================

cat("==========================================================\n")
cat("LOADING DATA VEHICLE COUNTS\n")
cat("==========================================================\n")

# Baca data dari CSV
data_raw <- read.csv("assets/file/vehicle_counts_processed.csv")

# Tampilkan struktur data
cat("\nStruktur data:\n")
str(data_raw)

# Pilih hanya kolom numerik untuk clustering (count_of_car, count_of_motorcycle, count_of_heavy)
x <- data_raw %>%
  dplyr::select(count_of_car, count_of_motorcycle, count_of_heavy) %>%
  as.matrix()

# Tampilkan statistik deskriptif
cat("\nStatistik Deskriptif:\n")
print(summary(x))

cat("\nDimensi data:", nrow(x), "baris,", ncol(x), "kolom\n")

# Opsional: Standardisasi data (uncomment jika diperlukan)
# x <- scale(x)

# Simpan data untuk digunakan
DataAsli <- x

cat("\n==========================================================\n")
cat("DATA SIAP UNTUK CLUSTERING\n")
cat("==========================================================\n\n")

# ==============================================================================
# 2. FCM (FUZZY C-MEANS) CLUSTERING
# ==============================================================================

cat("\n")
cat("==========================================================\n")
cat("FCM (FUZZY C-MEANS) CLUSTERING\n")
cat("==========================================================\n")

fcm_results <- list()

for(c in 2:10) {
  cat("\n--- Cluster k =", c, "---\n")

  # Initialize the memberships degrees matrix
  vu <- inaparc::imembrand(nrow(x), k=c)$u

  # Run FCM
  res.fcm <- fcm(x, center=c, memberships=vu)

  # Calculate Fuzzy Silhouette Index
  res.fcm1 <- ppclust2(res.fcm, "fclust")
  FSI <- SIL.F(res.fcm1$Xca, res.fcm1$U, alpha=1)

  cat("Fuzzy Silhouette Index =", FSI, "\n")

  # Simpan hasil
  fcm_results[[paste0("k", c)]] <- list(
    model = res.fcm,
    fsi = FSI
  )
}

cat("\nFCM SELESAI!\n")

# ==============================================================================
# 3. PCM (POSSIBILISTIC C-MEANS) CLUSTERING
# ==============================================================================

cat("\n")
cat("==========================================================\n")
cat("PCM (POSSIBILISTIC C-MEANS) CLUSTERING\n")
cat("==========================================================\n")

pcm_results <- list()

# PCM dengan random initialization
cat("\n--- PCM dengan Random Initialization ---\n")
for(c in 2:10) {
  cat("\n--- Cluster k =", c, "---\n")

  # Initialize the memberships degrees matrix
  vu <- inaparc::imembrand(nrow(x), k=c)$u

  # Run PCM
  res.pcm <- pcm(x, center=c, memberships=vu)

  # Calculate Fuzzy Silhouette Index
  res.pcm1 <- ppclust2(res.pcm, "fclust")
  FSI <- SIL.F(res.pcm1$Xca, res.pcm1$U, alpha=1)

  cat("Fuzzy Silhouette Index =", FSI, "\n")

  # Simpan hasil
  pcm_results[[paste0("random_k", c)]] <- list(
    model = res.pcm,
    fsi = FSI
  )
}

# PCM dengan K-means++ initialization
cat("\n--- PCM dengan K-means++ Initialization ---\n")
for(c in 2:10) {
  cat("\n--- Cluster k =", c, "---\n")

  # Initialize
  vu <- inaparc::imembrand(nrow(x), k=c)$u
  v <- inaparc::kmpp(x, k=c)$v

  # Run PCM
  res.pcm <- pcm(x, center=v, memberships=vu)

  # Calculate Fuzzy Silhouette Index
  res.pcm1 <- ppclust2(res.pcm, "fclust")
  FSI <- SIL.F(res.pcm1$Xca, res.pcm1$U, alpha=1)

  cat("Fuzzy Silhouette Index =", FSI, "\n")

  # Simpan hasil
  pcm_results[[paste0("kmpp_k", c)]] <- list(
    model = res.pcm,
    fsi = FSI
  )
}

# PCM dengan FCM initialization
cat("\n--- PCM dengan FCM Initialization ---\n")
for(c in 2:10) {
  cat("\n--- Cluster k =", c, "---\n")

  # Initialize
  vu <- inaparc::imembrand(nrow(x), k=c)$u
  Vc <- fcm(x, center=c, memberships=vu)$v

  # Run PCM
  res.pcm <- pcm(x, center=Vc, memberships=vu)

  # Calculate Fuzzy Silhouette Index
  res.pcm1 <- ppclust2(res.pcm, "fclust")
  FSI <- SIL.F(res.pcm1$Xca, res.pcm1$U, alpha=1)

  cat("Fuzzy Silhouette Index =", FSI, "\n")

  # Simpan hasil
  pcm_results[[paste0("fcm_k", c)]] <- list(
    model = res.pcm,
    fsi = FSI
  )
}

cat("\nPCM SELESAI!\n")

# ==============================================================================
# 4. FPCM (FUZZY POSSIBILISTIC C-MEANS) CLUSTERING
# ==============================================================================

cat("\n")
cat("==========================================================\n")
cat("FPCM (FUZZY POSSIBILISTIC C-MEANS) CLUSTERING\n")
cat("==========================================================\n")

fpcm_results <- list()

# FPCM dengan K-means++ initialization
cat("\n--- FPCM dengan K-means++ Initialization ---\n")
for(c in 2:10) {
  cat("\n--- Cluster k =", c, "---\n")

  # Initialize
  vu <- inaparc::imembrand(nrow(x), k=c)$u
  v <- inaparc::kmpp(x, k=c)$v

  # Run FPCM
  res.fpcm <- fpcm(x, center=v, memberships=vu)

  # Calculate Fuzzy Silhouette Index
  res.fpcm1 <- ppclust2(res.fpcm, "fclust")
  FSI <- SIL.F(res.fpcm1$Xca, res.fpcm1$U, alpha=1)

  cat("Fuzzy Silhouette Index =", FSI, "\n")

  # Simpan hasil
  fpcm_results[[paste0("kmpp_k", c)]] <- list(
    model = res.fpcm,
    fsi = FSI
  )
}

# FPCM dengan FCM initialization
cat("\n--- FPCM dengan FCM Initialization ---\n")
for(c in 2:10) {
  cat("\n--- Cluster k =", c, "---\n")

  # Initialize
  vu <- inaparc::imembrand(nrow(x), k=c)$u
  Vc <- fcm(x, center=c, memberships=vu)$v

  # Run FPCM
  res.fpcm <- fpcm(x, center=Vc, memberships=vu)

  # Calculate Fuzzy Silhouette Index
  res.fpcm1 <- ppclust2(res.fpcm, "fclust")
  FSI <- SIL.F(res.fpcm1$Xca, res.fpcm1$U, alpha=1)

  cat("Fuzzy Silhouette Index =", FSI, "\n")

  # Simpan hasil
  fpcm_results[[paste0("fcm_k", c)]] <- list(
    model = res.fpcm,
    fsi = FSI
  )
}

cat("\nFPCM SELESAI!\n")

# ==============================================================================
# 5. MFPCM (MODIFIED FUZZY POSSIBILISTIC C-MEANS) CLUSTERING
# ==============================================================================

cat("\n")
cat("==========================================================\n")
cat("MFPCM (MODIFIED FUZZY POSSIBILISTIC C-MEANS) CLUSTERING\n")
cat("==========================================================\n")

mfpcm_results <- list()

# MFPCM dengan random initialization
cat("\n--- MFPCM dengan Random Initialization ---\n")
for(c in 2:10) {
  cat("\n--- Cluster k =", c, "---\n")

  # Initialize
  vu <- inaparc::imembrand(nrow(x), k=c)$u

  # Run MFPCM
  res.mfpcm <- mfpcm(x, center=c, memberships=vu)

  # Calculate Fuzzy Silhouette Index
  res.mfpcm1 <- ppclust2(res.mfpcm, "fclust")
  FSI <- SIL.F(res.mfpcm1$Xca, res.mfpcm1$U, alpha=1)

  cat("Fuzzy Silhouette Index =", FSI, "\n")

  # Simpan hasil
  mfpcm_results[[paste0("random_k", c)]] <- list(
    model = res.mfpcm,
    fsi = FSI
  )
}

# MFPCM dengan K-means++ initialization
cat("\n--- MFPCM dengan K-means++ Initialization ---\n")
for(c in 2:10) {
  cat("\n--- Cluster k =", c, "---\n")

  # Initialize
  vu <- inaparc::imembrand(nrow(x), k=c)$u
  Vc <- inaparc::kmpp(x, k=c)$v

  # Run MFPCM
  res.mfpcm <- mfpcm(x, center=Vc, memberships=vu)

  # Calculate Fuzzy Silhouette Index
  res.mfpcm1 <- ppclust2(res.mfpcm, "fclust")
  FSI <- SIL.F(res.mfpcm1$Xca, res.mfpcm1$U, alpha=1)

  cat("Fuzzy Silhouette Index =", FSI, "\n")

  # Simpan hasil
  mfpcm_results[[paste0("kmpp_k", c)]] <- list(
    model = res.mfpcm,
    fsi = FSI
  )
}

# MFPCM dengan FCM initialization
cat("\n--- MFPCM dengan FCM Initialization ---\n")
for(c in 2:10) {
  cat("\n--- Cluster k =", c, "---\n")

  # Initialize
  vu <- inaparc::imembrand(nrow(x), k=c)$u
  Vc <- fcm(x, center=c, memberships=vu)$v

  # Run MFPCM
  res.mfpcm <- mfpcm(x, center=Vc, memberships=vu)

  # Calculate Fuzzy Silhouette Index
  res.mfpcm1 <- ppclust2(res.mfpcm, "fclust")
  FSI <- SIL.F(res.mfpcm1$Xca, res.mfpcm1$U, alpha=1)

  cat("Fuzzy Silhouette Index =", FSI, "\n")

  # Simpan hasil
  mfpcm_results[[paste0("fcm_k", c)]] <- list(
    model = res.mfpcm,
    fsi = FSI
  )
}

cat("\nMFPCM SELESAI!\n")

# ==============================================================================
# 6. VISUALISASI HASIL TERBAIK (K=3)
# ==============================================================================

cat("\n")
cat("==========================================================\n")
cat("VISUALISASI HASIL CLUSTERING (K=3)\n")
cat("==========================================================\n")

k_best <- 3

# FCM Visualization
cat("\n--- Visualisasi FCM (K=3) ---\n")
vu <- inaparc::imembrand(nrow(x), k=k_best)$u
res.fcm_best <- fcm(x, center=k_best, memberships=vu)
res.fcm_vis <- ppclust2(res.fcm_best, "kmeans")
print(fviz_cluster(res.fcm_vis, data=x, palette="jco", repel=TRUE,
                   main="FCM Clustering (K=3)"))

# PCM Visualization (FCM init)
cat("\n--- Visualisasi PCM dengan FCM Init (K=3) ---\n")
vu <- inaparc::imembrand(nrow(x), k=k_best)$u
v <- fcm(x, center=k_best, memberships=vu)$v
res.pcm_best <- pcm(x, center=v, memberships=vu)
res.pcm_vis <- ppclust2(res.pcm_best, "kmeans")
print(fviz_cluster(res.pcm_vis, data=x, palette="jco", repel=TRUE,
                   main="PCM Clustering (K=3)"))

# FPCM Visualization (K-means++ init)
cat("\n--- Visualisasi FPCM dengan K-means++ Init (K=3) ---\n")
vu <- inaparc::imembrand(nrow(x), k=k_best)$u
v <- inaparc::kmpp(x, k=k_best)$v
res.fpcm_best <- fpcm(x, center=v, memberships=vu)
res.fpcm_vis <- ppclust2(res.fpcm_best, "kmeans")
print(fviz_cluster(res.fpcm_vis, data=x, palette="jco", repel=TRUE,
                   main="FPCM Clustering (K=3)"))

# MFPCM Visualization (FCM init)
cat("\n--- Visualisasi MFPCM dengan FCM Init (K=3) ---\n")
vu <- inaparc::imembrand(nrow(x), k=k_best)$u
v <- fcm(x, center=k_best, memberships=vu)$v
res.mfpcm_best <- mfpcm(x, center=v, memberships=vu)
res.mfpcm_vis <- ppclust2(res.mfpcm_best, "kmeans")
print(fviz_cluster(res.mfpcm_vis, data=x, palette="jco", repel=TRUE,
                   main="MFPCM Clustering (K=3)"))

# ==============================================================================
# 7. RINGKASAN HASIL
# ==============================================================================

cat("\n")
cat("==========================================================\n")
cat("RINGKASAN FSI (FUZZY SILHOUETTE INDEX)\n")
cat("==========================================================\n")

# Fungsi untuk ekstrak FSI terbaik
get_best_fsi <- function(results_list, method_name) {
  fsi_values <- sapply(results_list, function(x) x$fsi)
  best_idx <- which.max(fsi_values)
  best_k <- as.numeric(gsub("\\D", "", names(results_list)[best_idx]))

  cat(sprintf("\n%s:\n", method_name))
  cat(sprintf("  Best K = %d\n", best_k))
  cat(sprintf("  Best FSI = %.4f\n", fsi_values[best_idx]))

  return(list(k=best_k, fsi=fsi_values[best_idx]))
}

# Tampilkan hasil terbaik
fcm_best <- get_best_fsi(fcm_results, "FCM")
pcm_random_best <- get_best_fsi(pcm_results[grep("random", names(pcm_results))], "PCM (Random)")
pcm_kmpp_best <- get_best_fsi(pcm_results[grep("kmpp", names(pcm_results))], "PCM (K-means++)")
pcm_fcm_best <- get_best_fsi(pcm_results[grep("fcm", names(pcm_results))], "PCM (FCM)")
fpcm_kmpp_best <- get_best_fsi(fpcm_results[grep("kmpp", names(fpcm_results))], "FPCM (K-means++)")
fpcm_fcm_best <- get_best_fsi(fpcm_results[grep("fcm", names(fpcm_results))], "FPCM (FCM)")
mfpcm_random_best <- get_best_fsi(mfpcm_results[grep("random", names(mfpcm_results))], "MFPCM (Random)")
mfpcm_kmpp_best <- get_best_fsi(mfpcm_results[grep("kmpp", names(mfpcm_results))], "MFPCM (K-means++)")
mfpcm_fcm_best <- get_best_fsi(mfpcm_results[grep("fcm", names(mfpcm_results))], "MFPCM (FCM)")

cat("\n")
cat("==========================================================\n")
cat("SEMUA CLUSTERING SELESAI!\n")
cat("==========================================================\n")
