function pwd_as_linux {
  "/$((pwd).Drive.Name.ToLowerInvariant())/$((pwd).Path.Replace('\', '/').Substring(3))"
}
docker run -it --name stock_auto --rm -v "$(pwd_as_linux)/../learning:/models/" -p 9041:9041 scpepper/graphpipe-tf:cpu1.13.1 --model=/models/stock_auto.pb --listen=0.0.0.0:9041
