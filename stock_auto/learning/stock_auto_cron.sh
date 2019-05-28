#!/bin/sh
cd /home/apps/apps/ml-timeseries-predict/stock_auto/learning
./get_daily_stock.sh
python stock_auto_client.py

exit 0
