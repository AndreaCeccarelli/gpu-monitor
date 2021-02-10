#!/bin/bash
LMS=5 #in un secondo:200 righe
LMSSupportedClock=1000 #senno logga l'inverosimile

echo "monitor query gpu"

# inforom.oem,inforom.ecc,inforom.power,\
nvidia-smi --format=csv -lms $LMS -f datalog/gpu.csv --query-gpu=timestamp,\
index,\
accounting.buffer_size,\
fan.speed,pstate,clocks_throttle_reasons.supported,clocks_throttle_reasons.active,\
clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.applications_clocks_setting,\
clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,\
clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown,\
clocks_throttle_reasons.sync_boost,memory.total,memory.used,memory.free,compute_mode,\
utilization.gpu,utilization.memory,encoder.stats.sessionCount,encoder.stats.averageFps,\
encoder.stats.averageLatency,ecc.mode.current,ecc.mode.pending,\
ecc.errors.corrected.volatile.device_memory,ecc.errors.corrected.volatile.register_file,\
ecc.errors.corrected.volatile.l1_cache,ecc.errors.corrected.volatile.l2_cache,\
ecc.errors.corrected.volatile.texture_memory,ecc.errors.corrected.volatile.total,\
ecc.errors.corrected.aggregate.device_memory,ecc.errors.corrected.aggregate.register_file,\
ecc.errors.corrected.aggregate.l1_cache,ecc.errors.corrected.aggregate.l2_cache,\
ecc.errors.corrected.aggregate.texture_memory,ecc.errors.corrected.aggregate.total,\
ecc.errors.uncorrected.volatile.device_memory,ecc.errors.uncorrected.volatile.register_file,\
ecc.errors.uncorrected.volatile.l1_cache,ecc.errors.uncorrected.volatile.l2_cache,\
ecc.errors.uncorrected.volatile.texture_memory,ecc.errors.uncorrected.volatile.total,\
ecc.errors.uncorrected.aggregate.device_memory,ecc.errors.uncorrected.aggregate.register_file,\
ecc.errors.uncorrected.aggregate.l1_cache,ecc.errors.uncorrected.aggregate.l2_cache,\
ecc.errors.uncorrected.aggregate.texture_memory,ecc.errors.uncorrected.aggregate.total,\
retired_pages.single_bit_ecc.count,retired_pages.double_bit.count,retired_pages.pending,\
temperature.gpu,power.management,power.draw,power.limit,enforced.power.limit,\
power.default_limit,power.min_limit,power.max_limit,clocks.current.graphics,\
clocks.current.sm,clocks.current.memory,clocks.current.video,clocks.applications.graphics,\
clocks.applications.memory,clocks.default_applications.graphics,clocks.default_applications.memory,\
clocks.max.graphics,clocks.max.sm,clocks.max.memory &

echo "monitor supported clocks"

#create final file for supported clocks
#echo "timestamp, memory [MHz], graphics [MHz]" | cat > supported-clocks-summary.csv

#start reading clocks values
nvidia-smi --format=csv -lms $LMSSupportedClock -f datalog/supported-clocks.csv --query-supported-clocks=timestamp,memory,graphics &

#takes everything from supported-clocks.csv and put in temporary file
#log: timestamp, sum of memory, sum of graphics
#tail -f supported-clocks.csv | awk -F"," '{z=$1; x+=$2; y+=$3}END{print z"," x"," y}' supported-clocks.csv >> supported-clocks-summary.csv


echo "monitor compute apps"

nvidia-smi --format=csv -lms $LMS -f datalog/compute-apps.csv --query-compute-apps=timestamp,pid,process_name,used_gpu_memory &

#echo "monitor accounted apps"
#nvidia-smi --format=csv -l $LMS -f accounted-apps.csv --query-accounted-apps=vgpu_instance,pid,gpu_utilization,mem_utilization,max_memory_usage,time &

echo "monitor retired pages"

nvidia-smi --format=csv -lms $LMS -f datalog/retired-pages.csv --query-retired-pages=timestamp,retired_pages.address,retired_pages.cause &
