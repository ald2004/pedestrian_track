cmake_minimum_required(VERSION 3.7)


project(CamEncode)

set(CMAKE_CXX_STANDARD 14)

include_directories(

         /usr/local/include/rockchip
)
	
link_directories(
	/usr/local/lib
                /usr/lib	
)
add_executable(CamEncode
CamEncode.cpp
main.cpp
)


target_link_libraries(CamEncode    pthread  
       rockchip_mpp

)


