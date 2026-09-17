// tokenizer.h — Emilia tokenizer port (EN path) + text normalizer.
//
// Ports zipvoice/tokenizer/tokenizer.py (EmiliaTokenizer) and
// zipvoice/tokenizer/normalizer.py (EnglishTextNormalizer).
// Self-contained G2P: espeak-ng (C library) replaces piper_phonemize.
#ifndef LUXTTS_TOKENIZER_H
#define LUXTTS_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>

#include "common.h"

namespace lux {

// EnglishTextNormalizer port: abbreviation expansion + number -> words
// (inflect number_to_words rules, dollars/cents, fractions, decimals,
// percent, ordinals).
std::string normalize_english_text(const std::string& text);

// map_punctuations port: fullwidth -> ASCII punctuation.
std::string map_punctuations(const std::string& text);

// Split text into language segments like get_segment(): returns (text, lang)
// where lang is one of "zh", "en", "pinyin", "tag", "other".
struct Segment { std::string text; std::string lang; };
std::vector<Segment> split_language_segments(const std::string& text);

struct Tokenizer {
  // tokens.txt: "<token>\t<id>" per line; pad token is "_" (id 0).
  std::unordered_map<std::string, int> token2id;
  int vocab_size = 0;
  int pad_id = 0;

  static Tokenizer load(const std::string& token_file, std::string* err);
  // Same parse from an in-memory table (the GGUF `luxtts.tokens` metadata
  // value — the self-contained model carries its own vocab).
  static Tokenizer load_str(const std::string& table, std::string* err);

  // texts_to_token_ids port: normalize -> segment -> G2P per segment ->
  // token ids (OOV skipped, like the Python).
  std::vector<int> texts_to_token_ids(const std::string& text) const;
};

// espeak-ng G2P: text + "en-us" -> phoneme token list (piper_phonemize
// equivalent: one token per phoneme, whitespace-separated tokens flattened).
std::vector<std::string> g2p_espeak_en_us(const std::string& text);

} // namespace lux

#endif // LUXTTS_TOKENIZER_H
