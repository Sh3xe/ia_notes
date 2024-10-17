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

	std::vector<uint8_t> data;
	uint32_t width;
	uint32_t height;
};

std::vector<Image> load_images( const std::string &file_path );

std::vector<uint8_t> load_labels( const std::string &file_path );