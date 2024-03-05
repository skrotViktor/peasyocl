find_package(OpenCL REQUIRED)

set(OPENCL_CLHPP_HEADERS_DIR .)

set(SOURCES DeformerContext.cpp)
set(HEADERS DeformerContext.h KernelUtils.h opencl.hpp)

add_library(${OCLMODULE_NAME}
    SHARED
        ${HEADERS}
        ${SOURCES}
)

target_link_libraries(${OCLMODULE_NAME}
    PUBLIC
        ${OpenCL_LIBRARIES}
)

set_target_properties(${OCLMODULE_NAME} PROPERTIES PREFIX "")

target_compile_features(${OCLMODULE_NAME}
    PUBLIC
        cxx_auto_type
)

install(
    FILES
        ${HEADERS}
    DESTINATION
        include/peasyocl
)

install(
    TARGETS ${OCLMODULE_NAME}
    EXPORT ${OCLMODULE_NAME}Targets
    LIBRARY DESTINATION lib
    INCLUDES DESTINATION include/peasyocl
)
install(
    EXPORT ${OCLMODULE_NAME}Targets
    DESTINATION cmake
)
