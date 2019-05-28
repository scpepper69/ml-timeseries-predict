#!/bin/sh
#target_stock=$1
#target_stock=7201
target_year=`date "+%Y"`
#target_date=`date "+%Y-%m-%d" -d -10days`
target_date=`date "+%Y-%m-%d" -d -1days`
history_data=/home/apps/apps/ml-timeseries-predict/stock_auto/learning/stock_auto_data.csv

get_daily_stock() {

  target_stock=${1}

  wget https://kabuoji3.com/stock/${target_stock}/${target_year}/ -O ${target_stock}.html 
  line=`grep -n ${target_date} ${target_stock}.html`
  line_cnt=`echo ${line} | cut -d ":" -f 1`
  close_line=`expr "${line_cnt}" + 4`
  if [[ $? -ne 0 ]]; then
    echo "There is no value today."
    exit 0
  else
    close_data=`awk "NR==${close_line}" ${target_stock}.html`
    close_val=`echo ${close_data} | sed -r 's/^.*>([0-9\.]+)<.*$/\1/'`
    echo ${close_val}
  fi

  rm -f ${target_stock}.html

  return 0

}

nissan=`get_daily_stock 7201`
toyota=`get_daily_stock 7203`
mazda=`get_daily_stock 7261`
honda=`get_daily_stock 7267`
subaru=`get_daily_stock 7270`

total="${nissa}n${toyot}a${mazda}${honda}${subaru}"
if [[ ${total} =~ "value" ]]; then
  echo "NG"
else  
  out_data=${target_date},${nissan},${toyota},${mazda},${honda},${subaru}
  if [[ `grep ${target_date} ${history_data} | wc -l` -eq 0 ]]; then
    echo ${out_data} >> ${history_data}
  else
    echo "The date value is already exists."
  fi
fi

exit 0
