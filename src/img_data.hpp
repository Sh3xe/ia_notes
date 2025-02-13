#pragma once

#include <vector>
#include <cstdint>
#include <string>

struct Image
{
	Image( std::vector<uint8_t> data, uint32_t width,	uint32_t height ):
		data(data), width(width), height(height)
	{
	}

	std::vector<double> convert_to_01_vector() const;

	std::vector<uint8_t> data;
	uint32_t width;
	uint32_t height;
};

/**
 * @brief Loads the MNIST images data in file_path
 * 
 * @param file_path 
 * @return std::vector<Image> 
 */
std::vector<Image> load_images( const std::string &file_path );

/**
 * @brief Loads the MNIST image labels located in file_path
 * 
 * @param file_path 
 * @return std::vector<uint8_t> 
 */
std::vector<uint8_t> load_labels( const std::string &file_path );