library(read.gt3x)
options(digits.secs = 3)
file = "PU6_CLE2B21130055_2017-03-28.gt3x"
exdir = tempdir()
files = unzip(file, exdir = exdir)
info_file = files[grepl("info", files)]
readLines(info_file)
readLines(files[grepl("log.bin", files)], n = 50)

info = extract_gt3x_info(info_file)

data = read.gt3x(file)
df = read.gt3x(file, asDataFrame = TRUE)

df$second = lubridate::floor_date(df$time, "seconds")
packets = split(df, df$second)
packets = unname(packets)
scale = info$`Acceleration Scale`
packet = packets[[5]]

function(
    df,
    sample_rate = 30L,
    max_g = c("8", "6")) {
  range_date = range(df$time, na.rm = TRUE)
  range_date = c(
    lubridate::floor_date(range_date[1], "seconds"),
    lubridate::ceiling_date(range_date[2], "seconds")
  )

  max_g = match.arg(max_g)
  serial_prefix = switch(
    max_g,
    "6" = "CLE2F",
    "8" = "MOS2F")
  acceleration_scale = switch(
    max_g,
    "6" = 341L,
    "8" = 256L)

  serial_number = paste0(serial_prefix,
                         paste(rep("0", 13-nchar(serial_prefix))))
  device_type = switch(
    max_g,
    "6" = "wGT3XPlus",
    "8" = "wGT3XBT")
  dynamic_range = as.integer(max_g)
  dynamic_range = c(-dynamic_range, dynamic_range)

  board_revision = switch(
    max_g,
    "6" = "1",
    "8" = "4"
  )
  unexpected_resets = "0"
  battery_voltage = "4.00"

  firmware = switch(
    max_g,
    "6" = "2.5.0",
    "8" = "1.9.2"
  )
  out = c(
    "Serial Number" = serial_number,
    "Device Type" = device_type,
    "Firmware" = firmware,
    "Battery Voltage" = battery_voltage,
    "Sample Rate" = sample_rate,
    # XXX,
    "Board Revision" = board_revision,
    "Unexpected Resets" = unexpected_resets,
    "Acceleration Scale" = sprintf("%.1f", acceleration_scale),
    "Acceleration Min" = sprintf("%.1f", dynamic_range[1]),
    "Acceleration Max" = sprintf("%.1f", dynamic_range[2]),

  )

  [6] "Start Date: 636380496000000000"
  [7] "Stop Date: 636386544000000000"
  [8] "Last Sample Time: 636386544000000000"
  [9] "TimeZone: 01:00:00"
  [10] "Download Date: 636397096570000000"
  [11] "Board Revision: 4"
  [12] "Unexpected Resets: 0"
  [13] "Acceleration Scale: 256.0"
  [14] "Acceleration Min: -8.0"
  [15] "Acceleration Max: 8.0"
  [16] "Limb: Wrist"
  [17] "Side: Right"
  [18] "Dominance: Dominant"
  [19] "Subject Name: P09"

}

create_packet = function(packet, scale = 341L) {

  payload = packet[, c("X", "Y", "Z")]
  packet_timestamp = unique(packet$second)
  packet_timestamp = as.integer(packet_timestamp)

  payload = round(payload * scale)
  payload = c(t(payload))
  payload = as.integer(payload)
  size = length(payload) * 2L # 2 bytes
  size = as.integer(size)
  checksum = 8L + size

  # writeBin("\x1e"
  header = c(
    # # log separator
    # writeBin("\x1e", raw(0), size=1L, endian="little"),
    # # ACTIVITY2 Packet
    # writeBin("\x1a", raw(0), size=1L, endian="little"),
    # log separator
    writeBin("\x1e\x1a", raw(0), size=2L, endian="little", useBytes = TRUE)[1:2],
    # timestamp
    # 1421142129L
    # writeBin(1490718600L, raw(0), size = 4L, endian = "little"),
    writeBin(packet_timestamp, raw(0), size = 4L, endian = "little"),
    # size in bytes
    writeBin(size, raw(0), size = 2L, endian = "little")[1:2],
    # payload/data
    writeBin(payload, raw(0), endian = "little", size = 2L),
    writeBin(checksum, raw(0), endian = "little", size = 1L)[1]
  )
}


header = pbapply::pblapply(packets, create_packet)
header = unname(header)
header = unlist(header)

file.copy(files[grepl("info", files)], "info.txt", overwrite = TRUE)
writeBin(header, con = "log.bin")
zip("test.gt3x", files = c("log.bin", "info.txt"))

c("Serial Number: CLE2B21130055", "Device Type: wGT3XPlus", "Firmware: 2.5.0",
  "Battery Voltage: 4.18", "Sample Rate: 30", "Start Date: 636263154000000000",
  "Stop Date: 636269202000000000", "Last Sample Time: 636269202000000000",
  "TimeZone: 01:00:00", "Download Date: 636270788020000000", "Board Revision: 1",
  "Unexpected Resets: 0", "Acceleration Scale: 341.0", "Acceleration Min: -6.0",
  "Acceleration Max: 6.0", "Sex: Male", "Limb: Wrist", "Side: Left",
  "Dominance: Non-Dominant", "Subject Name: P04")

out = read.gt3x("test.gt3x")

# path = here::here("csv_example")
# files = list.files(path = path, pattern = "^GT3", full.names = TRUE)
#
# df = vroom::vroom(files[1])
# rescale = function(x) round(x * 256)
# df = df %>%
#   mutate(across(c(X, Y, Z), rescale))
# data = c(48L, -52, 332)
# data=  rep(data, 100)
# con = tempfile(fileext = ".txt")
# data = as.integer(data)
# writeBin(data, con, endian = "little", size = 2)
# readFormat(con,
#            memFormat(integers=vectorBlock(integer2, 20)))
# # 0x1A
