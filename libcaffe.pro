TEMPLATE = lib
unix{
CONFIG(debug, debug|release): TARGET = caffed
CONFIG(release, debug|release): TARGET = caffe
}
win32{
CONFIG(debug, debug|release): TARGET = libcaffed
CONFIG(release, debug|release): TARGET = libcaffe
}
INCLUDEPATH += include src include/caffe/proto
DEFINES += USE_OPENCV CPU_ONLY
CONFIG += dll
#CONFIG += staticlib
DESTDIR += ../caffe/build_qt

# Input
HEADERS += include/caffe/common.hpp include/caffe/sgd_solvers.hpp include/caffe/filler.hpp include/caffe/syncedmem.hpp include/caffe/util/device_alternate.hpp include/caffe/util/blocking_queue.hpp include/caffe/util/insert_splits.hpp include/caffe/util/im2col.hpp include/caffe/util/db.hpp include/caffe/util/format.hpp include/caffe/util/upgrade_proto.hpp include/caffe/util/cudnn.hpp include/caffe/util/hdf5.hpp include/caffe/util/rng.hpp include/caffe/util/benchmark.hpp include/caffe/util/mkl_alternate.hpp include/caffe/util/io.hpp include/caffe/util/db_leveldb.hpp include/caffe/util/math_functions.hpp include/caffe/util/db_lmdb.hpp include/caffe/data_transformer.hpp include/caffe/solver.hpp include/caffe/data_reader.hpp include/caffe/layer.hpp include/caffe/layer_factory.hpp include/caffe/solver_factory.hpp include/caffe/net.hpp include/caffe/blob.hpp include/caffe/layers/cudnn_lcn_layer.hpp include/caffe/layers/data_layer.hpp include/caffe/layers/deconv_layer.hpp include/caffe/layers/flatten_layer.hpp include/caffe/layers/lstm_layer.hpp include/caffe/layers/dummy_data_layer.hpp include/caffe/layers/softmax_layer.hpp include/caffe/layers/memory_data_layer.hpp include/caffe/layers/batch_norm_layer.hpp include/caffe/layers/prelu_layer.hpp include/caffe/layers/inner_product_layer.hpp include/caffe/layers/hinge_loss_layer.hpp include/caffe/layers/reduction_layer.hpp include/caffe/layers/slice_layer.hpp include/caffe/layers/neuron_layer.hpp include/caffe/layers/tile_layer.hpp include/caffe/layers/cudnn_tanh_layer.hpp include/caffe/layers/elu_layer.hpp include/caffe/layers/embed_layer.hpp include/caffe/layers/bnll_layer.hpp include/caffe/layers/softmax_loss_layer.hpp include/caffe/layers/multinomial_logistic_loss_layer.hpp include/caffe/layers/threshold_layer.hpp include/caffe/layers/lrn_layer.hpp include/caffe/layers/input_layer.hpp include/caffe/layers/batch_reindex_layer.hpp include/caffe/layers/cudnn_conv_layer.hpp include/caffe/layers/split_layer.hpp include/caffe/layers/hdf5_data_layer.hpp include/caffe/layers/parameter_layer.hpp include/caffe/layers/infogain_loss_layer.hpp include/caffe/layers/scale_layer.hpp include/caffe/layers/pooling_layer.hpp include/caffe/layers/python_layer.hpp include/caffe/layers/absval_layer.hpp include/caffe/layers/argmax_layer.hpp include/caffe/layers/reshape_layer.hpp include/caffe/layers/cudnn_softmax_layer.hpp include/caffe/layers/image_data_layer.hpp include/caffe/layers/exp_layer.hpp include/caffe/layers/tanh_layer.hpp include/caffe/layers/sigmoid_layer.hpp include/caffe/layers/filter_layer.hpp include/caffe/layers/power_layer.hpp include/caffe/layers/contrastive_loss_layer.hpp include/caffe/layers/euclidean_loss_layer.hpp include/caffe/layers/crop_layer.hpp include/caffe/layers/bias_layer.hpp include/caffe/layers/cudnn_pooling_layer.hpp include/caffe/layers/spp_layer.hpp include/caffe/layers/cudnn_relu_layer.hpp include/caffe/layers/base_conv_layer.hpp include/caffe/layers/conv_layer.hpp include/caffe/layers/recurrent_layer.hpp include/caffe/layers/accuracy_layer.hpp include/caffe/layers/log_layer.hpp include/caffe/layers/silence_layer.hpp include/caffe/layers/hdf5_output_layer.hpp include/caffe/layers/cudnn_sigmoid_layer.hpp include/caffe/layers/mvn_layer.hpp include/caffe/layers/relu_layer.hpp include/caffe/layers/concat_layer.hpp include/caffe/layers/dropout_layer.hpp include/caffe/layers/im2col_layer.hpp include/caffe/layers/cudnn_lrn_layer.hpp include/caffe/layers/window_data_layer.hpp include/caffe/layers/eltwise_layer.hpp include/caffe/layers/base_data_layer.hpp include/caffe/layers/sigmoid_cross_entropy_loss_layer.hpp include/caffe/layers/rnn_layer.hpp include/caffe/layers/loss_layer.hpp include/caffe/internal_thread.hpp include/caffe/caffe.hpp include/caffe/parallel.hpp include/caffe/util/signal_handler.h include/caffe/proto/caffe.pb.h
SOURCES += src/caffe/layer.cpp src/caffe/layer_factory.cpp src/caffe/common.cpp src/caffe/net.cpp src/caffe/solver.cpp src/caffe/util/db_lmdb.cpp src/caffe/util/io.cpp src/caffe/util/db.cpp src/caffe/util/math_functions.cpp src/caffe/util/hdf5.cpp src/caffe/util/signal_handler.cpp src/caffe/util/upgrade_proto.cpp src/caffe/util/benchmark.cpp src/caffe/util/blocking_queue.cpp src/caffe/util/cudnn.cpp src/caffe/util/insert_splits.cpp src/caffe/util/im2col.cpp src/caffe/util/db_leveldb.cpp src/caffe/syncedmem.cpp src/caffe/data_reader.cpp src/caffe/data_transformer.cpp src/caffe/internal_thread.cpp src/caffe/layers/data_layer.cpp src/caffe/layers/contrastive_loss_layer.cpp src/caffe/layers/softmax_loss_layer.cpp src/caffe/layers/crop_layer.cpp src/caffe/layers/base_data_layer.cpp src/caffe/layers/dummy_data_layer.cpp src/caffe/layers/log_layer.cpp src/caffe/layers/recurrent_layer.cpp src/caffe/layers/lstm_unit_layer.cpp src/caffe/layers/conv_layer.cpp src/caffe/layers/lrn_layer.cpp src/caffe/layers/pooling_layer.cpp src/caffe/layers/silence_layer.cpp src/caffe/layers/rnn_layer.cpp src/caffe/layers/deconv_layer.cpp src/caffe/layers/cudnn_lrn_layer.cpp src/caffe/layers/memory_data_layer.cpp src/caffe/layers/dropout_layer.cpp src/caffe/layers/threshold_layer.cpp src/caffe/layers/softmax_layer.cpp src/caffe/layers/parameter_layer.cpp src/caffe/layers/cudnn_conv_layer.cpp src/caffe/layers/base_conv_layer.cpp src/caffe/layers/loss_layer.cpp src/caffe/layers/input_layer.cpp src/caffe/layers/batch_reindex_layer.cpp src/caffe/layers/embed_layer.cpp src/caffe/layers/cudnn_relu_layer.cpp src/caffe/layers/relu_layer.cpp src/caffe/layers/reshape_layer.cpp src/caffe/layers/hdf5_data_layer.cpp src/caffe/layers/cudnn_sigmoid_layer.cpp src/caffe/layers/euclidean_loss_layer.cpp src/caffe/layers/tile_layer.cpp src/caffe/layers/absval_layer.cpp src/caffe/layers/batch_norm_layer.cpp src/caffe/layers/filter_layer.cpp src/caffe/layers/inner_product_layer.cpp src/caffe/layers/multinomial_logistic_loss_layer.cpp src/caffe/layers/prelu_layer.cpp src/caffe/layers/exp_layer.cpp src/caffe/layers/concat_layer.cpp src/caffe/layers/bias_layer.cpp src/caffe/layers/power_layer.cpp src/caffe/layers/spp_layer.cpp src/caffe/layers/elu_layer.cpp src/caffe/layers/window_data_layer.cpp src/caffe/layers/cudnn_softmax_layer.cpp src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp src/caffe/layers/hdf5_output_layer.cpp src/caffe/layers/accuracy_layer.cpp src/caffe/layers/bnll_layer.cpp src/caffe/layers/image_data_layer.cpp src/caffe/layers/im2col_layer.cpp src/caffe/layers/sigmoid_layer.cpp src/caffe/layers/tanh_layer.cpp src/caffe/layers/cudnn_pooling_layer.cpp src/caffe/layers/hinge_loss_layer.cpp src/caffe/layers/slice_layer.cpp src/caffe/layers/lstm_layer.cpp src/caffe/layers/neuron_layer.cpp src/caffe/layers/cudnn_tanh_layer.cpp src/caffe/layers/argmax_layer.cpp src/caffe/layers/flatten_layer.cpp src/caffe/layers/reduction_layer.cpp src/caffe/layers/infogain_loss_layer.cpp src/caffe/layers/split_layer.cpp src/caffe/layers/mvn_layer.cpp src/caffe/layers/cudnn_lcn_layer.cpp src/caffe/layers/eltwise_layer.cpp src/caffe/layers/scale_layer.cpp src/caffe/solvers/rmsprop_solver.cpp src/caffe/solvers/adadelta_solver.cpp src/caffe/solvers/nesterov_solver.cpp src/caffe/solvers/adagrad_solver.cpp src/caffe/solvers/adam_solver.cpp src/caffe/solvers/sgd_solver.cpp src/caffe/parallel.cpp src/caffe/blob.cpp

unix{
#opencv
LIBS += -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

#glog
LIBS += -lglog

#boost
LIBS += -lboost_system -lboost_filesystem -lm

#gflags
LIBS += -lgflags

#protbuf
LIBS += -lprotobuf

#hdf5
LIBS += -lhdf5 -lhdf5_hl

#levelDB
LIBS += -lleveldb

#lmdb
LIBS += -llmdb

#openblas
LIBS += -lcblas -latlas
}

win32{

# opencv
PATH_OPENCV_INCLUDE   = "H:\3rdparty\OpenCV\opencv310\build\include"
PATH_OPENCV_LIBRARIES = "H:\3rdparty\OpenCV\opencv310\build\x64\vc12\lib"
VERSION_OPENCV        = 310
INCLUDEPATH += $${PATH_OPENCV_INCLUDE}
CONFIG(debug, debug|release){
LIBS += -L$${PATH_OPENCV_LIBRARIES} -lopencv_core$${VERSION_OPENCV}d
LIBS += -L$${PATH_OPENCV_LIBRARIES} -lopencv_highgui$${VERSION_OPENCV}d
LIBS += -L$${PATH_OPENCV_LIBRARIES} -lopencv_imgcodecs$${VERSION_OPENCV}d
LIBS += -L$${PATH_OPENCV_LIBRARIES} -lopencv_imgproc$${VERSION_OPENCV}d
}
CONFIG(release, debug|release){
LIBS += -L$${PATH_OPENCV_LIBRARIES} -lopencv_core$${VERSION_OPENCV}
LIBS += -L$${PATH_OPENCV_LIBRARIES} -lopencv_highgui$${VERSION_OPENCV}
LIBS += -L$${PATH_OPENCV_LIBRARIES} -lopencv_imgcodecs$${VERSION_OPENCV}
LIBS += -L$${PATH_OPENCV_LIBRARIES} -lopencv_imgproc$${VERSION_OPENCV}
}
#glog
INCLUDEPATH += H:\3rdparty\glog\include
LIBS += -LH:\3rdparty\glog\lib\x64\v120\Debug\dynamic -llibglog

#boost
INCLUDEPATH += H:\3rdparty\boost\boost_1_59_0
CONFIG(debug, debug|release): BOOST_VERSION = "-vc120-mt-gd-1_59"
CONFIG(release, debug|release): BOOST_VERSION = "-vc120-mt-1_59"
LIBS += -LH:\3rdparty\boost\boost_1_59_0\lib64-msvc-12.0 \
    -llibboost_system$${BOOST_VERSION} \
    -llibboost_date_time$${BOOST_VERSION} \
    -llibboost_filesystem$${BOOST_VERSION} \
    -llibboost_thread$${BOOST_VERSION} \
    -llibboost_regex$${BOOST_VERSION}

#gflags
INCLUDEPATH += H:\3rdparty\gflags\include
CONFIG(debug, debug|release): LIBS += -LH:\3rdparty\gflags\x64\v120\dynamic\Lib -lgflagsd
CONFIG(release, debug|release): LIBS += -LH:\3rdparty\gflags\x64\v120\dynamic\Lib -lgflags

#protobuf
INCLUDEPATH += H:\3rdparty\protobuf\include
CONFIG(debug, debug|release): LIBS += -LH:\3rdparty\protobuf\lib\x64\v120\Debug -llibprotobuf
CONFIG(release, debug|release): LIBS += -LH:\3rdparty\protobuf\lib\x64\v120\Release -llibprotobuf

# hdf5
INCLUDEPATH += H:\3rdparty\hdf5\include
LIBS += -LH:\3rdparty\hdf5\lib\x64 -lhdf5 -lhdf5_hl -lhdf5_tools -lhdf5_cpp

# levelDb
INCLUDEPATH += H:\3rdparty\LevelDB\include
CONFIG(debug, debug|release): LIBS += -LH:\3rdparty\LevelDB\lib\x64\v120\Debug -lLevelDb
CONFIG(release, debug|release): LIBS += -LH:\3rdparty\LevelDB\lib\x64\v120\Release -lLevelDb

# lmdb
INCLUDEPATH += H:\3rdparty\lmdb\include
CONFIG(debug, debug|release): LIBS += -LH:\3rdparty\lmdb\lib\x64 -llmdbD
CONFIG(release, debug|release): LIBS += -LH:\3rdparty\lmdb\lib\x64 -llmdb

#openblas
INCLUDEPATH += H:\3rdparty\openblas\include
LIBS += -LH:\3rdparty\openblas\lib\x64 -llibopenblas
}
