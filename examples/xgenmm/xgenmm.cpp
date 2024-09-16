/* 
08/18/2024 - Yutong - The file is adpated from examples/llava/llava.h in the llama.cpp repository.
*/


#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "base64.hpp"
#include "clip.h"
#include "common.h"
#include "llama.h"
#include "xgenmm.h"


struct tensor_from_gguf
{
    struct ggml_tensor  *data;
    struct ggml_context *ctx;
};


void print_tensor(ggml_tensor *tensor, const char *name = "", int verbosity = 0)
{
    if (tensor->ne[2] == 1)
    {
        printf("---> %s: (%ld, %ld)\n", name, tensor->ne[0], tensor->ne[1]);
    }
    else if (ggml_is_3d(tensor))
    {
        printf("---> %s: (%ld, %ld, %ld)\n", name, tensor->ne[0], tensor->ne[1], tensor->ne[2]);
    }
    else
    {
        printf("---> %s: (%ld, %ld, %ld, %ld)\n", name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    }
    if (verbosity == 1)
    {
        printf("*********************************************************************\n");
        if (tensor->ne[2] == 1)
        {
            const float *mat = (float *)tensor->data;
            int          dim0 = tensor->ne[1];
            int          dim1 = tensor->ne[0];
            if (dim0 < 6 && dim1 < 6)
            {
                for (int i = 0; i < dim0; i++)
                {
                    for (int j = 0; j < dim1; j++)
                    {
                        printf("%+.4f ", mat[i * dim1 + j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            else
            {
                for (int i = 0; i < std::min(dim0, 3); i++)
                {
                    for (int j = 0; j < std::min(dim1, 3); j++)
                    {
                        printf("%+.6f ", mat[i * dim1 + j]);
                    }
                    printf("... ");
                    for (int j = dim1 - 3; j < dim1; j++)
                    {
                        printf("%+.6f ", mat[i * dim1 + j]);
                    }
                    printf("\n");
                }
                if (dim0 > 3)
                {
                    printf("...................... omit ......................\n");
                    for (int i = dim0 - 3; i < dim0; i++)
                    {
                        for (int j = 0; j < std::min(dim1, 3); j++)
                        {
                            printf("%+.6f ", mat[i * dim1 + j]);
                        }
                        printf("... ");
                        for (int j = dim1 - 3; j < dim1; j++)
                        {
                            printf("%+.6f ", mat[i * dim1 + j]);
                        }
                        printf("\n");
                    }
                }
            }
        }
        else if (ggml_is_3d(tensor))
        {
            const float *data = (float *)tensor->data;
            int          dim0 = tensor->ne[2];
            int          dim1 = tensor->ne[1];
            int          dim2 = tensor->ne[0];
            if (dim0 < 6 && dim1 < 6 && dim2 < 6)
            {
                for (int i = 0; i < dim0; i++)
                {
                    printf("dim0 = %d\n", i);
                    for (int j = 0; j < dim1; j++)
                    {
                        for (int k = 0; k < dim2; k++)
                        {
                            printf("%+.6f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("\n");
                    }
                    printf("\n");
                }
                printf("\n");
            }
            else
            {
                for (int i = 0; i < std::min(dim0, 3); i++)
                {
                    printf("dim0 = %d\n", i);
                    for (int j = 0; j < std::min(dim1, 3); j++)
                    {
                        for (int k = 0; k < std::min(dim2, 3); k++)
                        {
                            printf("%+.6f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("... ");
                        for (int k = dim2 - 3; k < dim2; k++)
                        {
                            printf("%+.6f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("\n");
                    }
                    printf("........................\n");
                    for (int j = dim1 - 3; j < dim1; j++)
                    {
                        for (int k = 0; k < std::min(dim2, 3); k++)
                        {
                            printf("%+.6f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("... ");
                        for (int k = dim2 - 3; k < dim2; k++)
                        {
                            printf("%+.6f ", data[i * dim1 * dim2 + j * dim2 + k]);
                        }
                        printf("\n");
                    }
                    printf("---------------------------------------------------\n");
                }
                printf("\n");
            }
        }
    }
    printf("*********************************************************************\n");
    printf("\n");
}


bool load_tensor_from_file(const char *filename, tensor_from_gguf &tensor)
{
    struct gguf_init_params params = {
        /*.no_alloc   =*/false,
        /*.ctx        =*/&tensor.ctx,
    };
    gguf_context *ctx = gguf_init_from_file(filename, params);
    if (!ctx)
    {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }
    tensor.data = ggml_get_tensor(tensor.ctx, "data");

    return true;
}

// RGB uint8 image
struct clip_image_u8
{
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32
{
    int nx;
    int ny;

    std::vector<float> buf;
};

struct clip_image_grid_shape
{
    int first;
    int second;
};

/**
 * Selects the best resolution from a list of possible resolutions based on the original size.
 *
 * @param original_size The original size of the image in the format (width, height).
 * @param possible_resolutions A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
 * @return The best fit resolution in the format (width, height).
 */
static std::pair<int, int> select_best_resolution(const std::pair<int, int>              &original_size,
                                                  const std::vector<std::pair<int, int>> &possible_resolutions)
{
    int original_width = original_size.first;
    int original_height = original_size.second;

    std::pair<int, int> best_fit;
    int                 max_effective_resolution = 0;
    int                 min_wasted_resolution = std::numeric_limits<int>::max();

    for (const auto &resolution : possible_resolutions)
    {
        int   width = resolution.first;
        int   height = resolution.second;
        float scale =
            std::min(static_cast<float>(width) / original_width, static_cast<float>(height) / original_height);
        int downscaled_width = static_cast<int>(original_width * scale);
        int downscaled_height = static_cast<int>(original_height * scale);
        int effective_resolution = std::min(downscaled_width * downscaled_height, original_width * original_height);
        int wasted_resolution = (width * height) - effective_resolution;
        // LOG_TEE("resolution: %d %d, scale: %f, downscaled: %d %d, effective: %d, wasted: %d\n", width, height, scale,
        // downscaled_width, downscaled_height, effective_resolution, wasted_resolution);
        if (effective_resolution > max_effective_resolution ||
            (effective_resolution == max_effective_resolution && wasted_resolution < min_wasted_resolution))
        {
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
            best_fit = resolution;
        }
    }

    return best_fit;
}

/**
 * @brief Get the anyres image grid shape object
 *
 * @param image_size
 * @param grid_pinpoints
 * @param image_patch_size
 * @return <int, int>
 */
static struct clip_image_grid_shape get_anyres_image_grid_shape(const std::pair<int, int>              &image_size,
                                                                const std::vector<std::pair<int, int>> &grid_pinpoints,
                                                                int image_patch_size)
{
    /**
        Conversion from gguf flat array to vector:
        std::vector<std::pair<int, int>> possible_resolutions;
        for (int i = 0; i < 32 && params.image_grid_pinpoints[i] != 0; i+=2) {
            possible_resolutions.push_back({params.image_grid_pinpoints[i], params.image_grid_pinpoints[i+1]});
        }
     */
    auto best_resolution = select_best_resolution(image_size, grid_pinpoints);
    return {best_resolution.first / image_patch_size, best_resolution.second / image_patch_size};
}

// Take the image segments in a grid configuration and return the embeddings and the number of embeddings into
// preallocated memory (image_embd_out)
static bool clip_llava_handle_patches(clip_ctx *ctx_clip, std::vector<float *> &image_embd_v,
                                      struct clip_image_grid_shape grid_shape, float *image_embd_out,
                                      int *n_img_pos_out)
{
    struct
    {
        struct ggml_context *ctx;
    } model;

    const int32_t image_size = clip_image_size(ctx_clip);
    const int32_t patch_size = clip_patch_size(ctx_clip);

    int32_t num_patches_per_side =
        image_size / patch_size;  // 336 / 14 = 24 - used for embedding-patching boxes (24*24 = 576 patches)

    int num_patches_width = grid_shape.first;    // grid 1-4
    int num_patches_height = grid_shape.second;  // grid 1-4

    const size_t num_images = num_patches_width * num_patches_height + 1;

    // TODO: size calculation is not calculated - it's only tens of MB
    size_t ctx_size = 0;

    {
        ctx_size += clip_embd_nbytes(ctx_clip) * num_images * 8;  // image_features
        ctx_size += 1024 * 1024 * ggml_type_size(GGML_TYPE_F32);
    }

    struct ggml_init_params params
    {
        /*.mem_size   =*/ctx_size,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/false,  // NOTE: this should be false when using the legacy API
    };

    // Python reference code for full unpad:
    /*
        base_image_feature = image_feature[0]
        image_feature = image_feature[1:]
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = unpad_image(image_feature, image_sizes[image_idx])
        image_feature = torch.cat((
            image_feature,
            self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1)
        ), dim=-1)
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
    */
    // We now have two options: unpad or no unpad. Unpad removes tokens for faster llm eval.
    // In terms of result quality it appears to make no difference, so we'll start with the easier approach given 5D
    // tensors are not supported in ggml yet. Without unpad we have to split the sub-image embeddings into patches of 24
    // features each and permute them. Once all images are processed to prepended the base_image_features without any
    // changes.

    // Pytorch reference simplified, modified for ggml compatibility - confirmed identical output in python (for a 2x2
    // grid image (676x676 scaling))
    /*
        image_feature = image_feature.view(2, 2, 24, 24, 4096)
        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
        image_feature = image_feature.view(2, 24, 2, 24, 4096)
        image_feature = image_feature.flatten(0, 3)

        // Reshape to 4D tensor by merging the last two dimensions
        image_feature = image_feature.view(2, 2, 24, 24*4096)
        image_feature = image_feature.permute(0, 2, 1, 3).contiguous()
        image_feature = image_feature.view(-1, 4096)
    */

    model.ctx = ggml_init(params);

    struct ggml_tensor *image_features =
        ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, clip_n_mmproj_embd(ctx_clip), clip_n_patches(ctx_clip),
                           num_images - 1);  // example: 4096 x 576 x 4
    // ggml_tensor_printf(image_features,"image_features",__LINE__,false,false);
    // fill it with the image embeddings, ignoring the base
    for (size_t i = 1; i < num_images; i++)
    {
        size_t offset = (i - 1) * clip_embd_nbytes(ctx_clip);
        memcpy((uint8_t *)(image_features->data) + offset, image_embd_v[i], clip_embd_nbytes(ctx_clip));
    }

    struct ggml_cgraph *gf = ggml_new_graph(model.ctx);
    size_t              size_ele = ggml_type_size(GGML_TYPE_F32);

    struct ggml_tensor *image_features_patchview = ggml_view_4d(
        model.ctx, image_features, num_patches_per_side * clip_n_mmproj_embd(ctx_clip), num_patches_per_side,
        num_patches_width, num_patches_height, size_ele * num_patches_per_side * clip_n_mmproj_embd(ctx_clip),
        size_ele * num_patches_per_side * clip_n_mmproj_embd(ctx_clip) * num_patches_per_side,
        size_ele * num_patches_per_side * clip_n_mmproj_embd(ctx_clip) * num_patches_per_side * num_patches_width, 0);
    // ggml_tensor_printf(image_features_patchview,"image_features_patchview",__LINE__,false,false);
    struct ggml_tensor *permuted_cont =
        ggml_cont(model.ctx, ggml_permute(model.ctx, image_features_patchview, 0, 2, 1, 3));
    /**
     At the end of each row we have to add the row_end embeddings, which are the same as the newline embeddings
         image_feature = torch.cat((
        image_feature,
        self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
    ), dim=-1)
     *
     */

    // ggml_tensor_printf(permuted_cont,"permuted_cont",__LINE__,false,false);
    struct ggml_tensor *flatten =
        ggml_view_2d(model.ctx, permuted_cont, clip_n_mmproj_embd(ctx_clip),
                     num_patches_height * num_patches_width * num_patches_per_side * num_patches_per_side,
                     size_ele * clip_n_mmproj_embd(ctx_clip), 0);
    // ggml_tensor_printf(flatten,"flatten",__LINE__,false,false);
    ggml_build_forward_expand(gf, flatten);
    ggml_graph_compute_with_ctx(model.ctx, gf, 1);
    struct ggml_tensor *result = gf->nodes[gf->n_nodes - 1];

    memcpy(image_embd_out, image_embd_v[0], clip_embd_nbytes(ctx_clip));  // main image as global context
    // append without newline tokens (default behavior in llava_arch when not using unpad ):
    memcpy(image_embd_out + clip_n_patches(ctx_clip) * clip_n_mmproj_embd(ctx_clip), (float *)result->data,
           clip_embd_nbytes(ctx_clip) * (num_images - 1));  // grid patches
    *n_img_pos_out = static_cast<int>(result->ne[1] + clip_n_patches(ctx_clip));

    // Debug: Test single segments
    // Current findings: sending base image, sending a segment embedding all works similar to python
    // However, permuted embeddings do not work yet (stride issue?)
    // memcpy(image_embd_out, image_embd_v[0], clip_embd_nbytes(ctx_clip)); // main image as context
    // memcpy(image_embd_out, (float*)prepared_cont->data, clip_embd_nbytes(ctx_clip)); // main image as context
    // *n_img_pos_out=576;

    ggml_free(model.ctx);
    return true;
}

static bool clip_xgenmm_handle_vit_patches(clip_ctx *ctx_clip , const clip_image_u8 *img , std::vector<float *> &image_embd_v,
                                      struct clip_image_grid_shape grid_shape, float * image_embd)
                                      // float * image_embd: final output
{
    int original_width = img->nx;
    int original_height = img->ny;
    int num_images = image_embd_v.size();
    int32_t num_patches_per_side = clip_image_size(ctx_clip) / clip_patch_size(ctx_clip);
    int num_patches_width = grid_shape.first;
    int num_patches_height = grid_shape.second;
    int patch_num = num_patches_per_side * num_patches_per_side;  // 728
    int hidden_size = clip_hidden_size(ctx_clip); // 1152
    size_t  size_ele = ggml_type_size(GGML_TYPE_F32);

    struct
    {
        struct ggml_context* ctx;
    } model;

    // TODO: size calculation is not calculated - it's only tens of MB
    size_t ctx_size = 0;

    {
        ctx_size +=
            num_patches_per_side * num_patches_per_side * hidden_size * sizeof(float) * num_images * 8;  // image_features
        ctx_size += 1024 * 1024 * ggml_type_size(GGML_TYPE_F32);
    }
    struct ggml_init_params params
    {
        /*.mem_size   =*/ctx_size,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/false,  // NOTE: this should be false when using the legacy API
    };

    model.ctx = ggml_init(params);


    
    struct ggml_tensor* image_features = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, hidden_size, patch_num, num_images - 1); 
    struct ggml_tensor* base_image_feature = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, hidden_size, patch_num, 1);

    int dim0 = num_images - 1;
    int dim1 = num_patches_per_side * num_patches_per_side;
    int dim2 = hidden_size;

    float* image_features_data = (float*)image_features->data;
    float* base_image_feature_data = (float*)base_image_feature->data;

    for (int i=0; i < dim0; i++)
    {
        for (int j=0; j < dim1; j++)
        {
            for (int k=0; k < dim2; k++)
            {
                image_features_data[i * dim1 * dim2 + j * dim2 + k] =
                    image_embd_v[i+1][j * dim2 + k];
                if (i == 0)
                {
                    base_image_feature_data[j * dim2 + k] = image_embd_v[i][j * dim2 + k];
                }
            }
        }
    }


    struct ggml_tensor* image_features_patchview = ggml_view_4d(
        model.ctx, image_features, num_patches_per_side * hidden_size, num_patches_per_side,
        num_patches_width, num_patches_height, size_ele * num_patches_per_side * hidden_size,
        size_ele * num_patches_per_side * hidden_size * num_patches_per_side,
        size_ele * num_patches_per_side * hidden_size * num_patches_per_side * num_patches_width, 0);

    struct ggml_tensor* permuted_cont =
        ggml_cont(model.ctx, ggml_permute(model.ctx, image_features_patchview, 0, 2, 1, 3));

    struct ggml_tensor* flatten =
        ggml_view_2d(model.ctx, permuted_cont, hidden_size,
                     num_patches_height * num_patches_width * num_patches_per_side * num_patches_per_side,
                     size_ele * hidden_size, 0);
    
    struct ggml_tensor* tensor_3d =
    ggml_reshape_3d(model.ctx, flatten,
                    hidden_size,                                        
                    num_patches_per_side * num_patches_per_side, 
                    num_patches_width * num_patches_height);
    tensor_3d = ggml_cont(model.ctx, tensor_3d);
    tensor_3d =  ggml_concat(model.ctx, base_image_feature, tensor_3d, 2);
    struct ggml_cgraph* gf = ggml_new_graph(model.ctx);
    ggml_build_forward_expand(gf, tensor_3d);
    ggml_graph_compute_with_ctx(model.ctx, gf, 1);

    struct ggml_tensor* result = gf->nodes[gf->n_nodes - 1];
    print_tensor(result, "result after vit", 1);


    // {
    //     printf((" =========================     DEBUG  =========================\n"));
    //     printf("Load pre-computed image embeddings and attention_mask\n");
    //     std::string      filename = "/export/share/yutong/receipt_5patches_vision_features.gguf";
    //     tensor_from_gguf tensor;
    //     bool             is_successful = load_tensor_from_file(filename.c_str(), tensor);
    //     if (!is_successful)
    //     {
    //         fprintf(stderr, "%s: load_tensor_from_file() failed\n", __func__);
    //         return 1;
    //     }
    //     result = tensor.data;
    //     print_tensor(result, "load from pytorch", 1);
    //     // exit(1);
    //     // free(result);
    // }

    struct
    {
        struct ggml_context* ctx;
    } mask;

        ctx_size = 0;

    {
        ctx_size +=
            num_patches_per_side * num_patches_width * num_patches_per_side * num_patches_height * sizeof(float) * 4;
        ctx_size += 1024 * 1024 * ggml_type_size(GGML_TYPE_F32);
    }

    params = 
    {
        /*.mem_size   =*/ctx_size,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/false,  // NOTE: this should be false when using the legacy API
    };

    mask.ctx = ggml_init(params);
    int current_height = num_patches_per_side * num_patches_height;
    int current_width = num_patches_per_side * num_patches_width;

    float original_aspect_ratio = (float)original_width / (float)original_height;
    float current_aspect_ratio = (float)current_width / (float)current_height;
    // printf("original_height: %d, original_width: %d, original_aspect_ratio: %.2f\n", original_height, original_width,
    //        original_aspect_ratio);
    // printf("current_height: %d, current_width: %d, current_aspect_ratio: %.2f\n", current_height, current_width,
    //        current_aspect_ratio);
    float scale_factor = 1.0;
    struct ggml_tensor* attention_mask = ggml_new_tensor_2d(mask.ctx, GGML_TYPE_F32, current_width, current_height);
    float* attention_mask_data = (float*)attention_mask->data;
    if (original_aspect_ratio > current_aspect_ratio){
        scale_factor = (float)current_width / (float)original_width;
        int new_height = int(original_height * scale_factor);
        int padding = (current_height - new_height) / 2;
        // printf("new_height: %d, padding: %d\n", new_height, padding);
        
        for (int i = 0; i < current_height; i++){
            for (int j = 0; j < current_width; j++){
                if (i < padding || i >= current_height - padding)
                {
                    attention_mask_data[i * current_width + j] = 0.0;
                }
                else
                {
                    attention_mask_data[i * current_width + j] = 1.0;
                }
            }
        }
    }else{
        scale_factor = (float)current_height / (float)original_height;
        int new_width = int(original_width * scale_factor);
        int padding = (current_width - new_width) / 2;
        // printf("new_width: %d, padding: %d\n", new_width, padding);
        for (int i = 0; i < current_height; i++){
            for (int j = 0; j < current_width; j++){
                if (j < padding || j >= current_width - padding)
                {
                    attention_mask_data[i * current_width + j] = 0.0;
                }
                else
                {
                    attention_mask_data[i * current_width + j] = 1.0;
                }
            }
        }
    }

    attention_mask = ggml_reshape_2d(mask.ctx, attention_mask, num_patches_per_side * num_patches_per_side, num_patches_width * num_patches_height);
    attention_mask = ggml_cont(mask.ctx, attention_mask);
    struct ggml_tensor* all_one_tensor =
        ggml_new_tensor_2d(mask.ctx, GGML_TYPE_F32, num_patches_per_side * num_patches_per_side, 1);
    std::fill_n((float*)all_one_tensor->data, num_patches_per_side * num_patches_per_side, 1.0);
    attention_mask = ggml_concat(mask.ctx, all_one_tensor, attention_mask, 1);

    gf = ggml_new_graph(mask.ctx);
    ggml_build_forward_expand(gf, attention_mask);
    ggml_graph_compute_with_ctx(mask.ctx, gf, 1);
    attention_mask = gf->nodes[gf->n_nodes - 1];
    // memcpy(image_embd_v_m_mask_out, (float *)attention_mask->data, ggml_nbytes(attention_mask));

    

    // compute attnetion masks outside of the graph
    struct ggml_tensor * attn_bias_input;
    struct ggml_context * ctx0;
    if (attention_mask)
    {
        const int ctx_size = 1024 * 1024 * 1024;
        struct ggml_init_params params
        {
            /*.mem_size   =*/ctx_size,
                /*.mem_buffer =*/NULL,
                /*.no_alloc   =*/false,  // NOTE: this should be false when using the legacy API
        };
        ctx0 = ggml_init(params);
        // vision_attn_mask 
        // 1 -> 0
        // 0 -> -inf
        const int batch_size = attention_mask->ne[1];
        const int vision_seq_length = attention_mask->ne[0];
        for (int i = 0; i < batch_size * vision_seq_length; i++)
        {
            if (((float *)attention_mask->data)[i] == 1.0)
            {
                ((float *)attention_mask->data)[i] = 0.0;
            }
            else
            {
                ((float *)attention_mask->data)[i] = -INFINITY;
            }
        }
        const int lantents_seq_length = 128;  // lantents_seq_length
        struct ggml_tensor *all_zero_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, lantents_seq_length, batch_size);
        std::fill_n((float *)all_zero_tensor->data, lantents_seq_length * batch_size, 0.0);


        attention_mask = ggml_concat(ctx0, attention_mask, all_zero_tensor, 0);
        ggml_tensor *attn_bias = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, lantents_seq_length + vision_seq_length,
                                                    batch_size, lantents_seq_length);
        attn_bias = ggml_repeat(ctx0, attention_mask, attn_bias);
        attn_bias = ggml_cont(ctx0, ggml_permute(ctx0, attn_bias, 0, 2, 1, 3));

        struct ggml_cgraph *gf_temp = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf_temp, attn_bias);
        ggml_graph_compute_with_ctx(ctx0, gf_temp, 1);
        attn_bias_input = attn_bias;
    }
    int batch_size = num_patches_width * num_patches_height + 1;
    const bool encoded = clip_image_encode_tokenizer(
        ctx_clip, batch_size, result, attn_bias_input, image_embd);

    ggml_free(model.ctx);
    ggml_free(mask.ctx);
    return true;
}


static clip_image_f32 *only_v2_5_reshape_by_patch(clip_image_f32 *image, int patch_size)
{
    int             width = image->nx;
    int             height = image->ny;
    int             num_patches = (height / patch_size) * (width / patch_size);
    clip_image_f32 *patch = clip_image_f32_init();
    patch->nx = patch_size * num_patches;
    patch->ny = patch_size;
    patch->buf.resize(3 * patch->nx * patch->ny);

    int patch_index = 0;

    for (int i = 0; i < height; i += patch_size)
    {
        for (int j = 0; j < width; j += patch_size)
        {
            for (int pi = 0; pi < patch_size; ++pi)
            {
                for (int pj = 0; pj < patch_size; ++pj)
                {
                    int input_index = ((i + pi) * width + (j + pj)) * 3;
                    int output_index = (pi * patch_size * num_patches + patch_index * patch_size + pj) * 3;
                    patch->buf[output_index] = image->buf[input_index];
                    patch->buf[output_index + 1] = image->buf[input_index + 1];
                    patch->buf[output_index + 2] = image->buf[input_index + 2];
                }
            }
            patch_index++;
        }
    }
    return patch;
}

static bool encode_image_with_clip(clip_ctx *ctx_clip, int n_threads, const clip_image_u8 *img, float *image_embd,
                                   int *n_img_pos)
{
    // std::vector<clip_image_f32*> img_res_v; // format VectN x H x W x RGB (N x 336 x 336 x 3), so interleaved RGB -
    // different to the python implementation which is N x 3 x 336 x 336
    clip_image_f32_batch img_res_v;
    img_res_v.size = 0;
    img_res_v.data = nullptr;
    if (!clip_image_preprocess(ctx_clip, img, &img_res_v))
    {
        LOG_TEE("%s: unable to preprocess image\n", __func__);
        delete[] img_res_v.data;
        return false;
    }

    const int64_t t_img_enc_start_us = ggml_time_us();

    const char *mm_patch_merge_type = clip_patch_merge_type(ctx_clip);
    if (clip_is_minicpmv(ctx_clip))
    {
        std::vector<float *> image_embd_v;
        image_embd_v.resize(img_res_v.size);
        struct clip_image_size *load_image_size = clip_image_size_init();
        for (size_t i = 0; i < img_res_v.size; i++)
        {
            const int64_t t_img_enc_step_start_us = ggml_time_us();
            image_embd_v[i] = (float *)malloc(clip_embd_nbytes(ctx_clip));
            int patch_size = 14;
            load_image_size->width = img_res_v.data[i].nx;
            load_image_size->height = img_res_v.data[i].ny;
            clip_add_load_image_size(ctx_clip, load_image_size);
            bool encoded = false;
            int  has_minicpmv_projector = clip_is_minicpmv(ctx_clip);
            if (has_minicpmv_projector == 2)
            {
                encoded = clip_image_encode(
                    ctx_clip, n_threads, only_v2_5_reshape_by_patch(&img_res_v.data[i], patch_size), image_embd_v[i]);
            }
            else if (has_minicpmv_projector == 3)
            {
                encoded = clip_image_encode(ctx_clip, n_threads, &img_res_v.data[i], image_embd_v[i]);
            }
            if (!encoded)
            {
                LOG_TEE("Unable to encode image - spatial_unpad - subimage %d of %d\n", (int)i + 1,
                        (int)img_res_v.size);
                return false;
            }
            const int64_t t_img_enc_steop_batch_us = ggml_time_us();
            LOG_TEE("%s: step %d of %d encoded in %8.2f ms\n", __func__, (int)i + 1, (int)img_res_v.size,
                    (t_img_enc_steop_batch_us - t_img_enc_step_start_us) / 1000.0);
        }
        const int64_t t_img_enc_batch_us = ggml_time_us();
        LOG_TEE("%s: all %d segments encoded in %8.2f ms\n", __func__, (int)img_res_v.size,
                (t_img_enc_batch_us - t_img_enc_start_us) / 1000.0);

        int n_img_pos_out = 0;
        for (size_t i = 0; i < image_embd_v.size(); i++)
        {
            std::memcpy(image_embd + n_img_pos_out * clip_n_mmproj_embd(ctx_clip), image_embd_v[i],
                        clip_embd_nbytes(ctx_clip));
            n_img_pos_out += clip_n_patches(ctx_clip);
        }
        *n_img_pos = n_img_pos_out;
        for (size_t i = 0; i < image_embd_v.size(); i++)
        {
            free(image_embd_v[i]);
        }
        image_embd_v.clear();
        load_image_size->width = img->nx;
        load_image_size->height = img->ny;
        clip_add_load_image_size(ctx_clip, load_image_size);
        LOG_TEE("%s: load_image_size %d %d\n", __func__, load_image_size->width, load_image_size->height);
    }
    else if (clip_is_xgenmm(ctx_clip))
    {
        // Get image embedding right after VIT, merge before v tokenizer
        int n_img_pos_out = 0;  // # of output visual token
        std::vector<float *> image_embd_v;
        image_embd_v.resize(img_res_v.size);
        for (size_t i = 0; i < img_res_v.size; i++)
        {   
            n_img_pos_out += clip_n_patches(ctx_clip);

            // size_t allocated_size = clip_embd_nbytes(ctx_clip);
            const int vit_patch_num = clip_image_size(ctx_clip) / clip_patch_size(ctx_clip) * (clip_image_size(ctx_clip) / clip_patch_size(ctx_clip));
            image_embd_v[i] =
                (float *)malloc(vit_patch_num * clip_hidden_size(ctx_clip)* sizeof(float));  // If vit only, it should be 729 * 1152 * 4 = 3359232
            const bool encoded = clip_image_encode_vit(
                ctx_clip, n_threads, &img_res_v.data[i],
                image_embd_v[i]);
            if (!encoded)
            {
                LOG_TEE("Unable to encode image - spatial_unpad - subimage %d of %d\n", (int)i + 1,
                        (int)img_res_v.size);
                return false;
            }
            // for (int j = 0; j < 5; j++)
            // {
            //     printf("    %.4f ", image_embd_v[i][j]);
            // }
            // printf("\n");
        }

        *n_img_pos = n_img_pos_out;
        const int64_t t_img_enc_batch_us = ggml_time_us();
        LOG_TEE("%s: %d segments encoded in %8.2f ms\n", __func__, (int)img_res_v.size,
                (t_img_enc_batch_us - t_img_enc_start_us) / 1000.0);
        const int32_t *image_grid = clip_image_grid(ctx_clip);
        
        std::vector<std::pair<int, int>> grid_pinpoints;  //(384, 768) (768, 384) (768, 768) (1152, 384) (384, 1152)..
        for (int i = 0; i < 32 && image_grid[i] != 0; i += 2)
        {
            grid_pinpoints.push_back({image_grid[i], image_grid[i + 1]});
        }

        // free all img_res_v - not needed anymore
        delete[] img_res_v.data;
        img_res_v.size = 0;
        img_res_v.data = nullptr;

        const int32_t image_size = clip_image_size(ctx_clip);
        struct clip_image_grid_shape grid_shape =
            get_anyres_image_grid_shape({img->nx, img->ny}, grid_pinpoints, image_size);  // grid_shape.first is width (e.g., 3), grid_shape.second is height (e.g., 1)
        

        // patch merging + projection
        clip_xgenmm_handle_vit_patches(ctx_clip, img, image_embd_v, grid_shape, image_embd);
        for (size_t i = 0; i < image_embd_v.size(); i++)
        {
            free(image_embd_v[i]);
        }
        image_embd_v.clear();
    }
    else if (strcmp(mm_patch_merge_type, "spatial_unpad") != 0)
    {
        std::vector<float *> image_embd_v;
        image_embd_v.resize(img_res_v.size);
        for (size_t i = 0; i < img_res_v.size; i++)
        {
            image_embd_v[i] =
                (float *)malloc(clip_embd_nbytes(ctx_clip));  // 576 patches * 4096 embeddings * 4 bytes = 9437184
            const bool encoded = clip_image_encode(
                ctx_clip, n_threads, &img_res_v.data[i],
                image_embd_v[i]);  // image data is in 3x336x336 format and will be converted to 336x336x3 inside
            if (!encoded)
            {
                LOG_TEE("Unable to encode image - spatial_unpad - subimage %d of %d\n", (int)i + 1,
                        (int)img_res_v.size);
                return false;
            }
        }
        const int64_t t_img_enc_batch_us = ggml_time_us();
        LOG_TEE("%s: %d segments encoded in %8.2f ms\n", __func__, (int)img_res_v.size,
                (t_img_enc_batch_us - t_img_enc_start_us) / 1000.0);

        const int32_t *image_grid = clip_image_grid(ctx_clip);

        std::vector<std::pair<int, int>> grid_pinpoints;
        for (int i = 0; i < 32 && image_grid[i] != 0; i += 2)
        {
            grid_pinpoints.push_back({image_grid[i], image_grid[i + 1]});
        }

        // free all img_res_v - not needed anymore
        delete[] img_res_v.data;
        img_res_v.size = 0;
        img_res_v.data = nullptr;

        const int32_t image_size = clip_image_size(ctx_clip);

        struct clip_image_grid_shape grid_shape =
            get_anyres_image_grid_shape({img->nx, img->ny}, grid_pinpoints, image_size);

        int n_img_pos_out;
        clip_llava_handle_patches(ctx_clip, image_embd_v, grid_shape, image_embd, &n_img_pos_out);
        *n_img_pos = n_img_pos_out;

        for (size_t i = 0; i < image_embd_v.size(); i++)
        {
            free(image_embd_v[i]);
        }
        image_embd_v.clear();
    }
    else
    {
        // spatial_unpad llava-1.6 type embedding
        // TODO: CLIP needs batching support - in HF the llm projection is separate after encoding, which might be a
        // solution to quickly get batching working
        std::vector<float *> image_embd_v;
        image_embd_v.resize(img_res_v.size);
        for (size_t i = 0; i < img_res_v.size; i++)
        {
            image_embd_v[i] =
                (float *)malloc(clip_embd_nbytes(ctx_clip));  // 576 patches * 4096 embeddings * 4 bytes = 9437184
            const bool encoded = clip_image_encode(
                ctx_clip, n_threads, &img_res_v.data[i],
                image_embd_v[i]);  // image data is in 3x336x336 format and will be converted to 336x336x3 inside
            if (!encoded)
            {
                LOG_TEE("Unable to encode image - spatial_unpad - subimage %d of %d\n", (int)i + 1,
                        (int)img_res_v.size);
                return false;
            }
        }
        const int64_t t_img_enc_batch_us = ggml_time_us();
        LOG_TEE("%s: %d segments encoded in %8.2f ms\n", __func__, (int)img_res_v.size,
                (t_img_enc_batch_us - t_img_enc_start_us) / 1000.0);

        const int32_t *image_grid = clip_image_grid(ctx_clip);

        std::vector<std::pair<int, int>> grid_pinpoints;
        for (int i = 0; i < 32 && image_grid[i] != 0; i += 2)
        {
            grid_pinpoints.push_back({image_grid[i], image_grid[i + 1]});
        }

        // free all img_res_v - not needed anymore
        delete[] img_res_v.data;
        img_res_v.size = 0;
        img_res_v.data = nullptr;

        const int32_t image_size = clip_image_size(ctx_clip);

        struct clip_image_grid_shape grid_shape =
            get_anyres_image_grid_shape({img->nx, img->ny}, grid_pinpoints, image_size);

        int n_img_pos_out;
        clip_llava_handle_patches(ctx_clip, image_embd_v, grid_shape, image_embd, &n_img_pos_out);
        *n_img_pos = n_img_pos_out;

        for (size_t i = 0; i < image_embd_v.size(); i++)
        {
            free(image_embd_v[i]);
        }
        image_embd_v.clear();

        // debug image/segment/normalization content:
        // clip_image_u8 * tmp = clip_image_u8_init();
        // clip_image_convert_f32_to_u8(*image_feature, *tmp);
        // clip_image_save_to_bmp(*tmp, "image_feature.bmp");
    }
    LOG_TEE("%s: image embedding created: %d tokens\n", __func__, *n_img_pos);

    const int64_t t_img_enc_end_us = ggml_time_us();
    float         t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;

    LOG_TEE("\n%s: image encoded in %8.2f ms by CLIP (%8.2f ms per image patch)\n", __func__, t_img_enc_ms,
            t_img_enc_ms / *n_img_pos);

    return true;
}

bool llava_validate_embed_size(const llama_context *ctx_llama, const clip_ctx *ctx_clip)
{
    // make sure that the correct mmproj was used, i.e., compare apples to apples
    int  n_llama_embd = llama_n_embd(llama_get_model(ctx_llama));
    auto n_image_embd = clip_n_mmproj_embd(ctx_clip);
    if (n_image_embd != n_llama_embd)
    {
        LOG_TEE(
            "%s: embedding dim of the multimodal projector (%d) is not equal to that of LLaMA (%d). Make sure that you "
            "use the correct mmproj file.\n",
            __func__, n_image_embd, n_llama_embd);
        return false;
    }
    return true;
}

bool llava_image_embed_make_with_clip_img(clip_ctx *ctx_clip, int n_threads, const clip_image_u8 *img,
                                          float **image_embd_out, int *n_img_pos_out)
{
    int num_max_patches = 6;
    if (clip_is_minicpmv(ctx_clip))
    {
        num_max_patches = 10;
    }
    else if (clip_is_xgenmm(ctx_clip))
    {
        num_max_patches = 10;
    }
    float *image_embd =
        (float *)malloc(clip_embd_nbytes(ctx_clip) * num_max_patches);  // TODO: base on gridsize/llava model
    if (!image_embd)
    {
        LOG_TEE("Unable to allocate memory for image embeddings\n");
        return false;
    }

    int n_img_pos;  // 0
    if (!encode_image_with_clip(ctx_clip, n_threads, img, image_embd, &n_img_pos))
    {
        LOG_TEE("%s: cannot encode image, aborting\n", __func__);
        free(image_embd);
        return false;
    }
    *image_embd_out = image_embd;
    *n_img_pos_out = n_img_pos;

    return true;
}

bool llava_eval_image_embed(llama_context *ctx_llama, const struct llava_image_embed *image_embed, int n_batch,
                            int *n_past)
{
    int n_embd = llama_n_embd(llama_get_model(ctx_llama));

    for (int i = 0; i < image_embed->n_image_pos; i += n_batch)
    {
        int n_eval = image_embed->n_image_pos - i;
        if (n_eval > n_batch)
        {
            n_eval = n_batch;
        }
        llama_batch batch = {
            int32_t(n_eval),
            nullptr,
            (image_embed->embed + i * n_embd),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            *n_past,
            1,
            0,
        };
        if (llama_decode(ctx_llama, batch))
        {
            LOG_TEE("%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

struct llava_image_embed *llava_image_embed_make_with_bytes(struct clip_ctx *ctx_clip, int n_threads,
                                                            const unsigned char *image_bytes, int image_bytes_length)
{
    clip_image_u8 *img = clip_image_u8_init();
    if (!clip_image_load_from_bytes(image_bytes, image_bytes_length, img))
    {
        clip_image_u8_free(img);
        LOG_TEE("%s: can't load image from bytes, is it a valid image?", __func__);
        return NULL;
    }

    float *image_embed = NULL;
    int    n_image_pos = 0;
    bool   image_embed_result =
        llava_image_embed_make_with_clip_img(ctx_clip, n_threads, img, &image_embed, &n_image_pos);
    if (!image_embed_result)
    {
        clip_image_u8_free(img);
        LOG_TEE("%s: coulnd't embed the image\n", __func__);
        return NULL;
    }

    clip_image_u8_free(img);
    auto result = (llava_image_embed *)malloc(sizeof(llava_image_embed));
    result->embed = image_embed;
    result->n_image_pos = n_image_pos;
    return result;
}

static bool load_file_to_bytes(const char *path, unsigned char **bytesOut, long *sizeOut)
{
    auto file = fopen(path, "rb");
    if (file == NULL)
    {
        LOG_TEE("%s: can't read file %s\n", __func__, path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto buffer = (unsigned char *)malloc(fileSize);  // Allocate memory to hold the file data
    if (buffer == NULL)
    {
        LOG_TEE("%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, path);
        perror("Memory allocation error");
        fclose(file);
        return false;
    }
    errno = 0;
    size_t ret = fread(buffer, 1, fileSize, file);  // Read the file into the buffer
    if (ferror(file))
    {
        die_fmt("read error: %s", strerror(errno));
    }
    if (ret != (size_t)fileSize)
    {
        die("unexpectedly reached end of file");
    }
    fclose(file);  // Close the file

    *bytesOut = buffer;
    *sizeOut = fileSize;
    return true;
}

struct llava_image_embed *llava_image_embed_make_with_filename(struct clip_ctx *ctx_clip, int n_threads,
                                                               const char *image_path)
{
    unsigned char *image_bytes;
    long           image_bytes_length;
    auto           loaded = load_file_to_bytes(image_path, &image_bytes, &image_bytes_length);
    if (!loaded)
    {
        LOG_TEE("%s: failed to load %s\n", __func__, image_path);
        return NULL;
    }
    llava_image_embed *embed = llava_image_embed_make_with_bytes(ctx_clip, n_threads, image_bytes, image_bytes_length);
    free(image_bytes);

    return embed;
}

void llava_image_embed_free(struct llava_image_embed *embed)
{
    free(embed->embed);
    free(embed);
}
