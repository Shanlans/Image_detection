

def vgg_16_output_shape(width,height):
    def get_output_length(input_length):
        return input_length // 16

    return get_output_length(width), get_output_length(height)



