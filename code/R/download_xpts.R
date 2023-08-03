library(rvest)
library(curl)
library(dplyr)
library(here)
source("code/R/utils.R")


get_xpt(nh_table = "DEMO_Y")
get_xpt(nh_table = "DEMO_G")
get_xpt(nh_table = "DEMO_H")

get_xpt(nh_table = "PAXDAY_Y")
get_xpt(nh_table = "PAXDAY_G")
get_xpt(nh_table = "PAXDAY_H")

get_xpt(nh_table = "PAXMIN_Y")
get_xpt(nh_table = "PAXMIN_G")
get_xpt(nh_table = "PAXMIN_H")

