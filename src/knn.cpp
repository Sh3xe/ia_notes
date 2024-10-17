#include "knn.hpp"
#include <limits>

inline float abs( float x )
{
	return x > 0 ? x: -x;
}

inline float to_01( uint8_t x )
{
	return (float)x / 255.0;
}

float distance( const Image &img0, const Image &img1 )
{
	if( img0.height != img1.height || img0.width != img1.width )
		return 0.0f;
	
	float dst = 0.0f;

	for(size_t i = 0; i < img0.width; ++i)
	{
		for(size_t j = 0; j < img0.height; ++j)
		{
			dst += abs( to_01(img0.data[i+j*img0.width]) - to_01(img1.data[i+j*img0.width]) );
		}
	}

	return dst;
}

uint8_t knn( const Image &image, const std::vector<Image> &training_images, const std::vector<uint8_t> &training_labels )
{
	float min_dist = std::numeric_limits<float>::max();
	uint8_t label = 255;

	if( training_images.size() != training_labels.size() ) return 255;

	for( size_t i = 0; i < training_images.size(); ++i)
	{
		float dist = distance(image, training_images[i]);
		if(dist < min_dist)
		{
			min_dist = dist;
			label = training_labels[i];
		}
	}

	return label;
}