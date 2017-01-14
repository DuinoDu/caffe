#!/bin/bash

# 1. protoc
cd src/caffe/proto
protoc caffe.proto --cpp_out=.
cd ../../../
if [ ! -d include/caffe/proto ];then
    mkdir include/caffe/proto
fi
mv src/caffe/proto/caffe.pb.h include/caffe/proto/

# 2. find .h .hpp in include/caffe and .cpp in src/caffe
HEADERS='HEADERS += '
SOURCES='SOURCES += '
SPLIT=' '
for file in $(find include/caffe | grep '\.hpp$')
do
    if [[ $file =~ .*test_.* ]];then
        echo ignore $file
    else
        HEADERS=$HEADERS$SPLIT$file
    fi
done
for file in $(find include/caffe | grep '\.h$')
do
    if [[ $file =~ .*test_.* ]];then
        echo ignore $file
    else
        HEADERS=$HEADERS$SPLIT$file
    fi
done

for file in $(find src/caffe | grep '\.cpp$')
do
    if [[ $file =~ .*test_.* ]];then
        echo ignore $file
    else
        SOURCES=$SOURCES$SPLIT$file
    fi
done

for file in $(find src/caffe | grep '\.cc$')
do
    if [[ $file =~ .*test_.* ]];then
        echo ignore $file
    else
        SOURCES=$SOURCES$SPLIT$file
    fi
done

# 3. set HEADERS SOURCES in libcaffe.pro
#echo $HEADERS
#echo $SOURCES
find -name 'libcaffe.pro' | xargs perl -pi -e "s|HEADERS.+$|$HEADERS|g"
find -name 'libcaffe.pro' | xargs perl -pi -e "s|SOURCES.+$|$SOURCES|g"

echo 'Save to libcaffe.pro'
