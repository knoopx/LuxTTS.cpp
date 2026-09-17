// vocos.h — Vocos 48k dual-path vocoder port (linacodec + upstream vocos).
//
// Ports (verified against the Python sources):
//   linacodec vocos.py L38 (Vocos) — dual-path decode (backbone -> upsampler ->
//   head_48k @48k; head @24k resampled to 48k; Linkwitz-Riley crossover merge)
//   linacodec upsampler_block.py L98 (UpSamplerBlock: ConvTranspose1d L115 +
//   ResnetBlock L36), L21 Snake1d, L62 Normalize (GroupNorm 32 groups)
//   linacodec linkwitz.py L4 (crossover_merge_linkwitz_riley: rfft merge,
//   cubic-Hermite 8-bin transition)
//   upstream vocos (gemelo-ai) models.py L26 (VocosBackbone: ConvNeXt modules.py
//   L8), heads.py L26 (ISTFTHead, out L40), spectral_ops.py L7 (ISTFT
//   center=True: torch.hann_window, NOLA normalization, win/2 trim)
//
// Config (vocoder/config.yaml, all shapes verified against vocos.bin at load):
//   backbone:  VocosBackbone(in=100, dim=512, intermediate=1536, layers=8)
//   upsampler: in=512, factors=[2,1], kernels=[8,8] -> (512->256->128)
//   head:      ISTFTHead(dim=512, n_fft=1024, hop=256, win=1024, center)
//   head_48k:  same ISTFTHead
//   crossover: cutoff = freq_range = 12000 (luxvoice.py L36 overrides the 4000
//              default), sr=48000, transition 8 bins
//
// Tensor names bound in load() (vocos.bin keys kept verbatim in the GGUF):
//   backbone.embed.{weight,bias}            (512,100,7)
//   backbone.norm.{weight,bias}             (512,)  (LayerNorm, eps 1e-6)
//   backbone.convnext.{0..7}.dwconv.{weight,bias}  (512,1,7)
//   backbone.convnext.{0..7}.norm.{weight,bias}    (512,)
//   backbone.convnext.{0..7}.pwconv1.{weight,bias} (1536,512)
//   backbone.convnext.{0..7}.pwconv2.{weight,bias} (512,1536)
//   backbone.convnext.{0..7}.gamma          (512,)
//   backbone.final_layer_norm.{weight,bias}  (512,)
//   upsampler.upsample_layers.{0,1}.{weight,bias} (256,512,8)/(128,256,8)
//   upsampler.resnet_blocks.{0,1}.norm1.{weight,bias} (256,)/(128,)
//   upsampler.resnet_blocks.{0,1}.snake1.alpha       (256,)/(128,)
//   upsampler.resnet_blocks.{0,1}.conv1.{weight,bias} (256,256,3)/(128,128,3)
//   upsampler.resnet_blocks.{0,1}.norm2.{weight,bias}
//   upsampler.resnet_blocks.{0,1}.snake2.alpha
//   upsampler.resnet_blocks.{0,1}.conv2.{weight,bias}
//   upsampler.out_proj.{weight,bias}         (512,128)
//   upsampler.final_snake.alpha               (512,)
//   head.out.{weight,bias}                   (1026,512)
//   head_48k.out.{weight,bias}               (1026,512)
//
// The 48k path is the returned audio (return_48k=True default; the "smooth"
// mode return_48k=False == decode_48k_raw).
//
// Heavy 1-D convolutions are plain C++ loops (the ggml engine is used by the
// zipformer; the conv/upsampler math here is small-kernel and easier to
// verify in C++). Weights are copied out of the ggml tensors at load(), so
// the caller's context may be freed afterwards.

#ifndef LUXTTS_VOCOS_H
#define LUXTTS_VOCOS_H

#include <string>
#include <vector>

#include "common.h"

struct ggml_context;
struct ggml_tensor;

namespace lux {

class Vocos {
 public:
  Vocos() = default;

  // Bind the vocoder weights from the named GGUF tensors (names WITHOUT the
  // instance prefix, as listed above). Hard error on any missing/malformed
  // tensor. Copies the weights out of `named` (the ggml context is not
  // retained).
  std::string load(ggml_context* ctx,
                   const std::vector<std::pair<std::string, ggml_tensor*>>& named);

  // decode (1, 100, T) row-major -> 48 kHz mono wav (dual path + crossover).
  std::vector<float> decode(const float* feats, int T) const;

  // The "smooth" mode (return_48k=False): the 24k head resampled to 48k,
  // no crossover merge.
  std::vector<float> decode_48k_raw(const float* feats, int T) const;

  // True once the weights are bound.
  bool loaded() const { return loaded_; }

 private:
  // 24 kHz -> 48 kHz (integer 2x: linear upsample + Hann-windowed-sinc
  // lowpass; approximates torchaudio.transforms.Resample closely).
  std::vector<float> resample_24k_to_48k(const std::vector<float>& in) const;
  // linkwitz.py L4 (crossover_merge_linkwitz_riley) port: rfft merge, cutoff_bin = n_bins/2 (12000 Hz @ 48k),
  // 8-bin cubic-Hermite transition, merged = s1*m + s2*(1-m), irfft.
  std::vector<float> crossover_merge(const std::vector<float>& hi,
                                     const std::vector<float>& lo) const;

  // shared internals
  std::vector<float> backbone(const float* feats, int T) const;          // (512, T)
  std::vector<float> upsampler(const float* feats, int T) const;        // (512, 2T+1)
  // head path: (C, T) -> istft(center) -> audio @ head rate
  std::vector<float> istft_head(const std::vector<float>& feats, int T,
                                bool head_48k) const;

  // bound weights (copied out of the ggml context at load)
  struct W {
    // backbone
    std::vector<float> emb_w, emb_b;                 // (512,100,7),(512,)
    std::vector<float> bn_w, bn_b;                   // backbone.norm (512,)
    std::vector<float> fnorm_w, fnorm_b;             // final_layer_norm (512,)
    struct CNX { std::vector<float> dw_w, dw_b, n_w, n_b, p1_w, p1_b, p2_w, p2_b, gamma; };
    std::vector<CNX> cnx;                            // 8 blocks
    // upsampler
    std::vector<float> up0_w, up0_b, up1_w, up1_b;   // (256,512,8),(128,256,8)
    struct Resnet { std::vector<float> n1_w, n1_b, s1_a, c1_w, c1_b, n2_w, n2_b, s2_a, c2_w, c2_b; };
    std::vector<Resnet> resnet;                      // 2 blocks
    std::vector<float> oproj_w, oproj_b, fsnake_a;
    // heads
    std::vector<float> head_out_w, head_out_b;       // (1026,512)
    std::vector<float> head48_out_w, head48_out_b;  // (1026,512)
  } w_;
  bool loaded_ = false;
};

} // namespace lux

#endif // LUXTTS_VOCOS_H
