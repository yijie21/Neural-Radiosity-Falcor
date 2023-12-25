add_executable(train_radiosity
	train.cu
	../dependencies/tinyexr/tinyexr.h
	../dependencies/filesystem/fwd.h
	../dependencies/filesystem/path.h
	../dependencies/filesystem/directory.h
	../dependencies/filesystem/resolver.h
)
target_link_libraries(train_radiosity PUBLIC ${CUDA_LIBRARIES} ${TINYEXR_EXT_LIBRARIES} tiny-cuda-nn)
