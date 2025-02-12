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

/**
 * @brief Given a digit between 0 and 9, one hot encode into a vector of 10 element between 0.0 and 1.0
 * 
 * @param digit 
 * @return std::vector<double> 
 */
std::vector<double> one_hot_encode( uint8_t digit );