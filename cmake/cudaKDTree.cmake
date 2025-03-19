if(TARGET cudaKDTree)
    return()
endif()

include(FetchContent)

message(STATUS "Fetching cudaKDTree")

FetchContent_Declare(
    cudaKDTree
    GIT_REPOSITORY https://github.com/ingowald/cudaKDTree.git
    GIT_TAG e38fb539a2e378ef9be839d667042d8c43ff3f68
)
FetchContent_MakeAvailable(cudaKDTree)