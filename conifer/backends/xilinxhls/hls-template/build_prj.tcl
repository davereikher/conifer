#################
#    HLS4ML
#################
array set opt {
    reset      0
    csim       1
    synth      1
    cosim      0
    export     0
    vsynth     0
}

foreach arg $::argv {
  foreach o [lsort [array names opt]] {
    regexp "$o=+(\\w+)" $arg unused opt($o)
  }
}

file mkdir tb_data
set CSIM_RESULTS "./tb_data/csim_results.log"

if {$opt(reset)} {
    open_project -reset myproject_prj
} else {
    open_project myproject_prj
}

set_top myproject
add_files firmware/BDT.h
add_files firmware/myproject.cpp
add_files -tb myproject_test.cpp -cflags "-I firmware/"
add_files -tb firmware/weights
add_files -tb tb_data
if {$opt(reset)} {
    open_solution -reset "solution1"
} else {
    open_solution "solution1"
}

open_solution -reset "solution1"
set_part {xc7vx690tffg1927-2}
create_clock -period 5 -name default
set_clock_uncertainty 3
config_rtl -reset none

if {$opt(csim)} {
    csim_design
}

if {$opt(synth)} {
    csynth_design
}

if {$opt(cosim)} {
    cosim_design -trace_level all
}

if {$opt(export)} {
    export_design -format ip_catalog
}

if {$opt(vsynth)} {
    puts "NOT IMPLEMENTED YET"
}
exit
