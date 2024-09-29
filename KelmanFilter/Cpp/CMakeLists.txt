set(EIGEN3_INCLUDE_DIR "D:/Libraries/eigen-3.4.0/eigen-3.4.0")
aux_source_directory(source Kelman_DIRS)
add_library(KelmanCpp STATIC ${Kelman_DIRS})
target_include_directories(KelmanCpp PUBLIC include ${EIGEN3_INCLUDE_DIR})
