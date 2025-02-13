#include "img_data.hpp"
#include <fstream>

std::vector<double> Image::convert_to_01_vector() const
{
	std::vector<double> vec;
	vec.reserve(width*height);

	for(uint32_t i = 0; i < width*height; ++i)
	{
		vec.push_back((double)data[i] / 255.0f);
	}

	return std::move(vec);
}

uint32_t endian_swap( uint32_t num )
{
	return ((num & 0xff000000) >> 24) | ((num & 0x00ff0000) >> 8) | ((num & 0x0000ff00) << 8) | (num << 24);
}

std::vector<Image> load_images( const std::string &file_path )
{
	std::fstream file {file_path, std::ios::in | std::ios::binary };

	if( !file ) 
	{
		throw std::runtime_error("Cannot open " + file_path);
	}

	// Fetching the header data
	uint32_t image_count = 0, magic_number = 0;
	size_t width = 0, height = 0;
	file.read( (char*)&magic_number, 4);
	file.read( (char*)&image_count, 4);
	file.read( (char*)&height, 4 );
	file.read( (char*)&width, 4 );

	// We switch the endianess if needed
	if( magic_number != 2051) 
	{
		image_count = endian_swap(image_count);
		magic_number = endian_swap(magic_number);
		height = endian_swap(height);
		width = endian_swap(width);
	}

	if( file.eof() || file.fail() || magic_number != 2051 ) return {};

	std::vector<Image> images;
	images.reserve(image_count);

	std::vector<uint8_t> current_image_data;
	current_image_data.reserve(width*height);

	// Pixels...
	for( size_t i = 0; i < image_count; ++i )
	{
		current_image_data.clear();
		uint8_t byte = 0;
		for(size_t j = 0; j < width * height; ++j)
		{
			file.read( (char*)&byte, 1 );
			current_image_data.push_back(byte);
		}

		if( file.eof() || file.fail() ) return {};
		images.emplace_back(current_image_data, width, height);
	}

	return images;
}

std::vector<uint8_t> load_labels( const std::string &file_path )
{
	std::fstream file {file_path, std::ios::in | std::ios::binary };

	if( !file ) 
	{
		throw std::runtime_error("Cannot open " + file_path);
	}

	// Fetching header
	uint32_t label_count = 0, magic_number = 0;
	file.read( (char*)&magic_number, 4);
	file.read( (char*)&label_count, 4);

	// Switch the endianness if needed
	if( magic_number != 2049) 
	{
		magic_number = endian_swap(magic_number);
		label_count = endian_swap(label_count);
	}

	if( file.eof() || file.fail() || magic_number != 2049 ) return {};

	std::vector<uint8_t> labels;
	labels.reserve(label_count);

	// Get the labels...
	for( size_t i = 0; i < label_count; ++i )
	{
		uint8_t byte = 0;
		file.read((char*)&byte, 1);

		if( file.eof() || file.fail() ) return {};
		labels.push_back(byte);
	}

	return labels;
}
