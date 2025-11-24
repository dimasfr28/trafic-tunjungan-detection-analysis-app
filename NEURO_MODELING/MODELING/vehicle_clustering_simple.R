# ==============================================================================
# FUZZY CLUSTERING UNTUK VEHICLE COUNTS DATA - SIMPLIFIED
# FCM, PCM, FPCM, MFPCM dengan K=3
# ==============================================================================

# Load required libraries
library(dplyr)
library(ppclust)
library(fclust)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

cat("==========================================================\n")
cat("FUZZY CLUSTERING - K=3\n")
cat("==========================================================\n\n")

# Baca data dari CSV
data_raw <- read.csv("assets/file/vehicle_counts_processed.csv")
x <- data_raw %>%
  dplyr::select(count_of_car, count_of_motorcycle, count_of_heavy) %>%
  as.matrix()

cat("Data:", nrow(x), "rows,", ncol(x), "columns\n\n")

# Set k=3
k <- 3

# Initialize memberships
set.seed(42)
vu <- inaparc::imembrand(nrow(x), k=k)$u

# ==============================================================================
# 2. JALANKAN SEMUA METODE
# ==============================================================================

results <- list()

# FCM
cat("Running FCM...\n")
res.fcm <- fcm(x, center=k, memberships=vu)
res.fcm1 <- ppclust2(res.fcm, "fclust")
fsi.fcm <- SIL.F(res.fcm1$Xca, res.fcm1$U, alpha=1)
results$FCM <- list(model=res.fcm, fsi=fsi.fcm, clusters=res.fcm$cluster)
cat("FCM - Fuzzy Silhouette Index:", fsi.fcm, "\n\n")

# PCM
cat("Running PCM...\n")
res.pcm <- pcm(x, center=k, memberships=vu)
res.pcm1 <- ppclust2(res.pcm, "fclust")
fsi.pcm <- SIL.F(res.pcm1$Xca, res.pcm1$U, alpha=1)
results$PCM <- list(model=res.pcm, fsi=fsi.pcm, clusters=res.pcm$cluster)
cat("PCM - Fuzzy Silhouette Index:", fsi.pcm, "\n\n")

# FPCM
cat("Running FPCM...\n")
res.fpcm <- fpcm(x, center=k, memberships=vu)
res.fpcm1 <- ppclust2(res.fpcm, "fclust")
fsi.fpcm <- SIL.F(res.fpcm1$Xca, res.fpcm1$U, alpha=1)
results$FPCM <- list(model=res.fpcm, fsi=fsi.fpcm, clusters=res.fpcm$cluster)
cat("FPCM - Fuzzy Silhouette Index:", fsi.fpcm, "\n\n")

# MFPCM
cat("Running MFPCM...\n")
res.mfpcm <- mfpcm(x, center=k, memberships=vu)
res.mfpcm1 <- ppclust2(res.mfpcm, "fclust")
fsi.mfpcm <- SIL.F(res.mfpcm1$Xca, res.mfpcm1$U, alpha=1)
results$MFPCM <- list(model=res.mfpcm, fsi=fsi.mfpcm, clusters=res.mfpcm$cluster)
cat("MFPCM - Fuzzy Silhouette Index:", fsi.mfpcm, "\n\n")

# ==============================================================================
# 3. PILIH METODE TERBAIK
# ==============================================================================

cat("==========================================================\n")
cat("HASIL PERBANDINGAN\n")
cat("==========================================================\n\n")

fsi_scores <- c(FCM=fsi.fcm, PCM=fsi.pcm, FPCM=fsi.fpcm, MFPCM=fsi.mfpcm)
print(fsi_scores)

best_method <- names(which.max(fsi_scores))
best_fsi <- max(fsi_scores)

cat("\n==========================================================\n")
cat("METODE TERBAIK:", best_method, "\n")
cat("FSI:", best_fsi, "\n")
cat("==========================================================\n\n")

# ==============================================================================
# 4. EXTRACT HASIL TERBAIK DAN URUTKAN CLUSTER
# ==============================================================================

best_result <- results[[best_method]]
best_clusters <- best_result$clusters

# Hitung rata-rata total kendaraan untuk setiap cluster
cluster_density <- data.frame(
  original_cluster = 1:k,
  mean_total = numeric(k)
)

for(i in 1:k) {
  cluster_idx <- which(best_clusters == i)
  cluster_density$mean_total[i] <- mean(
    data_raw$count_of_car[cluster_idx] +
    data_raw$count_of_motorcycle[cluster_idx] +
    data_raw$count_of_heavy[cluster_idx]
  )
}

# Urutkan cluster berdasarkan kepadatan (rendah ke tinggi)
cluster_density <- cluster_density[order(cluster_density$mean_total), ]
cluster_density$new_cluster <- 1:k

cat("\n==========================================================\n")
cat("MAPPING CLUSTER (BERDASARKAN KEPADATAN)\n")
cat("==========================================================\n")
print(cluster_density)
cat("\n")

# Remap cluster labels
cluster_mapping <- setNames(cluster_density$new_cluster, cluster_density$original_cluster)
best_clusters_sorted <- cluster_mapping[as.character(best_clusters)]

# Tambahkan cluster labels ke data original
data_raw$cluster <- best_clusters_sorted
data_raw$method <- best_method

# Simpan hasil
output_file <- "assets/file/vehicle_counts_fuzzy_clustered.csv"
write.csv(data_raw, output_file, row.names=FALSE)

cat("Hasil clustering disimpan ke:", output_file, "\n\n")

# Tampilkan ringkasan cluster
cat("==========================================================\n")
cat("RINGKASAN CLUSTER\n")
cat("==========================================================\n\n")

for(i in 1:k) {
  cluster_data <- data_raw[data_raw$cluster == i, ]
  cat(sprintf("Cluster %d: %d data points\n", i, nrow(cluster_data)))
  cat("  Mean count_of_car:", mean(cluster_data$count_of_car), "\n")
  cat("  Mean count_of_motorcycle:", mean(cluster_data$count_of_motorcycle), "\n")
  cat("  Mean count_of_heavy:", mean(cluster_data$count_of_heavy), "\n\n")
}

cat("==========================================================\n")
cat("SELESAI!\n")
cat("==========================================================\n")
