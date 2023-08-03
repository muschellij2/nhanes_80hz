
#' Count Steps According to Gu et al, 2017 Method
#'
#' This method is based off finding peaks in the summed and squared acceleration signal
#' and then using multiple thresholds to determine if each peak is a step or an artifact.
#' An additional magnitude threshold was added to the algorithm to prevent false positives
#' in free living data.
#' @param input_data An `n x 3` matrix or data.frame of accelerometry values
#' @param sample_rate sampling frequency of the input data
#' @param k window size for controlling peak finding.
#' @param periodicity_range a length-2 vector of the range of periodicity.
#' These are integers that represent **samples** not seconds.
#' @param similarity_threshold threshold (in g) for similarity between
#' magnitude of peaks
#' @param continuity_window_size Window size for continuity
#' @param continuity_threshold Threshold for continuity
#' @param variance_threshold Variance threshold for the signal
#' @param vm_threshold vector magnitude threshold for a peak to be
#' called a peak
#'
#' @return A vector of length `round(nrow(input_data) / sample_rate)` of the
#' estimated steps, where the data is rounded to seconds
#' @export
#'
#' @examples
#' input_data <- matrix(runif(500 * 3, min = -1.5, max = 1.5), ncol = 3)
#' verisense_count_steps(input_data, sample_rate = 15L)
verisense_count_steps <- function(
    input_data,
    sample_rate,
    k = 3, # window size
    periodicity_range = c(5, 15),
    similarity_threshold = -0.5,
    continuity_window_size = 4,
    continuity_threshold = 3,
    variance_threshold = 0.001,
    vm_threshold = 1.2
) {

  assertthat::assert_that(
    assertthat::is.count(sample_rate),
    assertthat::is.scalar(k),
    ncol(input_data) == 3,
    length(periodicity_range) == 2,
    assertthat::is.scalar(similarity_threshold),
    assertthat::is.count(continuity_window_size),
    assertthat::is.count(continuity_threshold),
    assertthat::is.scalar(variance_threshold),
    assertthat::is.scalar(vm_threshold)
  )
  periodicity_range = sort(periodicity_range)
  assertthat::assert_that(
    diff(periodicity_range) > 0
  )

  acc <- sqrt(input_data[, 1]^2 + input_data[, 2]^2 + input_data[, 3]^2)
  n_samples = length(acc)
  if (sd(acc) < 0.025) {
    # acceleration too low, no steps
    num_seconds <- round(n_samples / sample_rate)
    steps_per_sec <- rep(0, num_seconds)
    return(steps_per_sec)
  }

  period_min = periodicity_range[1]
  period_max = periodicity_range[2]
  # Search for steps
  # Thresholds

  # find the peak rms value is every range of k
  half_k <- round(k / 2)
  segments <- floor(n_samples / k)
  peak_info <- matrix(NA, nrow = segments, ncol = 5)
  colnames(peak_info) <- c(
    "peak_location",
    "acc_magnitude",
    "periodicity",
    "similarity",
    "continuity"
  )
  peak_info <- as.data.frame(peak_info)
  # peak_info[,1] - peak location
  # peak_info[,2] - acc magnitude
  # peak_info[,3] - periodicity (samples)
  # peak_info[,4] - similarity
  # peak_info[,5] - continuity

  # for each segment find the peak location
  for (i in 1:segments) {
    start_idx <- (i - 1) * k + 1
    end_idx <- start_idx + (k - 1)
    tmp_loc_a <- which.max(acc[start_idx:end_idx])
    tmp_loc_b <- (i - 1) * k + tmp_loc_a
    # only save if this is a peak value in range of -k/2:+K/2
    start_idx_ctr <- tmp_loc_b - half_k
    if (start_idx_ctr < 1) {
      start_idx_ctr <- 1
    }
    end_idx_ctr <- tmp_loc_b + half_k
    if (end_idx_ctr > n_samples) {
      end_idx_ctr <- n_samples
    }
    check_loc <- which.max(acc[start_idx_ctr:end_idx_ctr])
    if (check_loc == (half_k + 1)) {
      peak_info[i, "peak_location"] <- tmp_loc_b
      peak_info[i, "acc_magnitude"] <- max(acc[start_idx:end_idx])
    }
  }
  peak_info <- peak_info[!is.na(peak_info[, "peak_location"]), ] # get rid of na rows

  # filter peak_info[,2] based on vm_threshold
  peak_info <- peak_info[peak_info[, "acc_magnitude"] > vm_threshold, ]
  no_steps <- FALSE
  if (nrow(peak_info) > 2) { # there must be at least two steps
    num_peaks <- nrow(peak_info)
    # Calculate Features (periodicity, similarity, continuity)
    peak_info[1:(num_peaks - 1), "periodicity"] <- diff(peak_info[, "peak_location"]) # calculate periodicity
    na_periodicity = is.na(peak_info[, "periodicity"])
    peak_info <- peak_info[
      !na_periodicity &
        peak_info[, "periodicity"] > period_min &
        peak_info[, "periodicity"] < period_max, ] # filter peaks based on period_min
  } else {
    no_steps <- TRUE
  }

  if (nrow(peak_info) == 0 || no_steps == TRUE) {
    # no steps found
    num_seconds <- round(n_samples / sample_rate)
    steps_per_sec <- rep(0, num_seconds)
    return(steps_per_sec)
  }

  # calculate similarity
  num_peaks <- nrow(peak_info)
  peak_info[1:(num_peaks - 2), "similarity"] <- -abs(diff(peak_info[, "acc_magnitude"], 2)) # calculate similarity
  peak_info = peak_info[!is.na(peak_info$similarity), ]
  peak_info <- peak_info[peak_info[, "similarity"] > similarity_threshold, ] # filter based on sim_thres

  # calculate continuity
  peak_info[, "continuity"] = 0
  if (nrow(peak_info) > 5) {
    end_for <- nrow(peak_info) - 1
    for (i in continuity_window_size:end_for) {
      # for each bw peak period calculate acc var
      v_count <- 0 # count how many windows were over the variance threshold
      for (x in 1:continuity_window_size) {
        index = peak_info[i - x + 1, "peak_location"]:peak_info[i - x + 2, "peak_location"]
        sub_acc = acc[index]
        if (var(sub_acc) > variance_threshold) {
          v_count <- v_count + 1
        }
      }
      if (v_count >= continuity_threshold) {
        peak_info[i, "continuity"] <- 1 # set continuity to 1, otherwise, 0
      }
    }
  }
  peak_info <- peak_info[peak_info[, "continuity"] == 1, ] # continuity test - only keep locations after this
  all_peak_info = peak_info
  peak_info = peak_info$peak_location
  peak_info <- peak_info[!is.na(peak_info)] # previous statement can result in an NA in col-1

  if (length(peak_info) == 0) {
    # no steps found
    num_seconds <- round(n_samples / sample_rate)
    steps_per_sec <- rep(0, num_seconds)
    return(steps_per_sec)
  } else {
    # for GGIR, output the number of steps in 1 second chunks
    start_idx_vec <- seq(from = 1, to = n_samples, by = sample_rate)
    steps_per_sec <- table(factor(findInterval(peak_info, start_idx_vec),
                                  levels = seq_along(start_idx_vec)))
    steps_per_sec <- as.numeric(steps_per_sec)
  }


  return(steps_per_sec)
}



