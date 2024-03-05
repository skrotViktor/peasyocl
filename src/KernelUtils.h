#ifndef OCL_KERNEL_UTILS
#define OCL_KERNEL_UTILS

#define OCL_KERNEL_PATHS_ENVIRONMENT "OCL_KERNEL_PATHS"

#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <vector>

namespace peasyocl::utils {

/**
 * @brief Load data of file to an std::string
 * 
 * @param path 
 * @return std::string 
 */
inline std::string LoadClKernelSource(const std::string_view& path) {
    static std::string content;
	std::ifstream filestream(path);

	if (!filestream.is_open()) {
		std::cout<< "Error: Failed to load file: " << path << std::endl;
	}
	std::string line = "";
	while(!filestream.eof()) {
		std::getline(filestream, line);
		content.append(line +"\n");
	}
    return content;
}

/**
 * @brief Find all paths in OCL_KERNEL_PATHS environment variable
 * 
 * @return const std::vector<std::string> 
 */
inline const std::vector<std::string> GetKernelPaths() {
    std::vector<std::string> result;
    const char* env = std::getenv(OCL_KERNEL_PATHS_ENVIRONMENT);

    if (!env) {
        std::cout<<"Failed: environment variable not set: " << OCL_KERNEL_PATHS_ENVIRONMENT << std::endl;
        return result;
    }

    std::string stringEnv(env);
    size_t prev = 0;
    size_t found = stringEnv.find(":");
    if (found == std::string::npos) {
        result.push_back(stringEnv);
        return result;
    }

    while(found != std::string::npos) {
        std::string substr = stringEnv.substr(prev, found);
        std::cout<<substr<<std::endl;
        result.push_back(substr);
        prev = found;
        found = stringEnv.find(":", found+1);
    }
    return result;
}

/**
 * @brief Get the Cl File By Name object
 * 
 * @param name Name of cl file. The file should be visible in the OCL_KERNEL_PATHS environment variable
 * @return const std::string 
 */
inline const std::string GetClFileByName(const std::string& name) {
    std::vector<std::string> paths = GetKernelPaths();
    if (paths.size() == 0) {
        std::cout<<"Failed: Could not find kernel paths"<< std::endl;
        return "";
    }

    for (auto& it: paths) {
        std::string path = it + "/" + name;
        if (name.rfind(".cl") == std::string::npos) {
            std::ifstream file(path + ".cl");
            if (file.good()) {
                return path;
            }
        }
        if (name.rfind(".ocl") == std::string::npos) {
            std::ifstream file(path + ".ocl");
            if (file.good()) {
                return path;
            }
        }
        std::ifstream file(path);
        if (file.good()) {
            return path;
        }
    }
    return "";
}

}

#endif