#!/bin/bash
#Our custom function
cust_func(){
echo "Doing split $1 ..."

python3 collecting_images.py $1
  sleep 1
}
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do
cust_func $i  &
done