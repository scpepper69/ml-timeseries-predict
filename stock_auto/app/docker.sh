docker run -d -it --name stock_auto --rm -v "$PWD/../learning:/models/" -p 9041:9041 scpepper/graphpipe-tf:cpu1.13.1 --model=/models/stock_auto.pb --listen=0.0.0.0:9041
