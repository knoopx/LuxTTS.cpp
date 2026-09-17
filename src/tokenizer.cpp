// tokenizer.cpp — zipvoice/tokenizer/tokenizer.py (EmiliaTokenizer) and
// zipvoice/tokenizer/normalizer.py L17-160 (EnglishTextNormalizer).
//
// G2P: zipvoice/tokenizer/tokenizer.py L158-166 (EspeakTokenizer.g2p,
// piper_phonemize).
#include "tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>

namespace lux {

namespace {

// normalizer.py L87-152 (character classes for the number passes)
inline bool isdigit8(unsigned char c) { return c >= '0' && c <= '9'; }

// normalizer.py L26 (re \w word boundaries)
inline bool is_word_char(unsigned char c) {
  return isdigit8(c) || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

// normalizer.py L87-152 + tokenizer.py L369-474 (character/UTF-8 helpers)
inline bool is_ascii_alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

// tokenizer.py L369-474 (UTF-8 helpers for map_punctuations/get_segment)
uint32_t utf8_next(const std::string& s, size_t& i) {
  size_t n = s.size();
  if (i >= n) return 0xFFFD;
  unsigned char b0 = (unsigned char)s[i];
  uint32_t cp = 0;
  int len = 1;
  if (b0 < 0x80) cp = b0;
  else if ((b0 & 0xE0) == 0xC0) { cp = b0 & 0x1F; len = 2; }
  else if ((b0 & 0xF0) == 0xE0) { cp = b0 & 0x0F; len = 3; }
  else if ((b0 & 0xF8) == 0xF0) { cp = b0 & 0x07; len = 4; }
  else { ++i; return 0xFFFD; }
  if (i + (size_t)len > n) { ++i; return 0xFFFD; }
  i++;
  for (int k = 1; k < len; k++) {
    unsigned char b = (unsigned char)s[i];
    if ((b & 0xC0) != 0x80) { i++; return 0xFFFD; }
    cp = (cp << 6) | (b & 0x3F);
    i++;
  }
  return cp;
}

// tokenizer.py L369-474 (UTF-8 helpers for map_punctuations/get_segment)
size_t utf8_codepoint_len(const std::string& s, size_t i) {
  if (i >= s.size()) return 0;
  unsigned char b0 = (unsigned char)s[i];
  if (b0 < 0x80) return 1;
  if ((b0 & 0xE0) == 0xC0) return 2;
  if ((b0 & 0xF0) == 0xE0) return 3;
  if ((b0 & 0xF8) == 0xF0) return 4;
  return 1;
}

// normalizer.py L155-159 + tokenizer.py L369-386 (string replacement helpers)
std::string string_replace_all(const std::string& s, const std::string& from, const std::string& to) {
  if (from.empty()) return s;
  std::string out;
  size_t last = 0, pos;
  while ((pos = s.find(from, last)) != std::string::npos) {
    out.append(s, last, pos - last);
    out += to;
    last = pos + from.size();
  }
  out.append(s, last, std::string::npos);
  return out;
}

[[maybe_unused]]
// tokenizer.py L369-474 (UTF-8 encode for the G2P token stream)
std::string utf8_encode(uint32_t cp) {
  std::string out;
  if (cp < 0x80) out += (char)cp;
  else if (cp < 0x800) {
    out += (char)(0xC0 | (cp >> 6));
    out += (char)(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += (char)(0xE0 | (cp >> 12));
    out += (char)(0x80 | ((cp >> 6) & 0x3F));
    out += (char)(0x80 | (cp & 0x3F));
  } else {
    out += (char)(0xF0 | (cp >> 18));
    out += (char)(0x80 | ((cp >> 12) & 0x3F));
    out += (char)(0x80 | ((cp >> 6) & 0x3F));
    out += (char)(0x80 | (cp & 0x3F));
  }
  return out;
}

const char* const kUnit[] = {"zero", "one", "two", "three", "four", "five", "six",
                             "seven", "eight", "nine"};
const char* const kTeen[] = {"ten", "eleven", "twelve", "thirteen", "fourteen",
                             "fifteen", "sixteen", "seventeen", "eighteen",
                             "nineteen"};
const char* const kTen[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty",
                            "seventy", "eighty", "ninety"};
const char* const kScale[] = {"", "thousand", "million", "billion", "trillion",
                              "quadrillion", "quintillion", "sextillion",
                              "septillion", "octillion", "nonillion", "decillion"};

// normalizer.py L124-143 (_expand_number: inflect number_to_words)
std::string group0_group_words(int v, const char* andword) {
  int h = v / 100, rem = v % 100;
  std::string s;
  if (h) { s += kUnit[h]; s += " hundred"; }
  if (rem) {
    std::string rw;
    if (rem < 10) rw = kUnit[rem];
    else if (rem < 20) rw = kTeen[rem - 10];
    else {
      int t = rem / 10, o = rem % 10;
      rw = kTen[t];
      if (o) { rw += "-"; rw += kUnit[o]; }
    }
    if (h) s += (andword && andword[0]) ? std::string(" ") + andword + " " : " ";
    s += rw;
  }
  return s;
}

// normalizer.py L124-143 (_expand_number: inflect number_to_words)
std::string num_words_group0(uint64_t n, const char* andword, const char* zero) {
  if (n == 0) return std::string(zero);
  int groups[12];
  int ng = 0;
  uint64_t m = n;
  while (m > 0) {
    groups[ng++] = (int)(m % 1000);
    m /= 1000;
  }
  std::vector<std::string> parts;
  for (int i = ng - 1; i >= 0; --i) {
    int v = groups[i];
    if (v == 0) continue;
    std::string w = group0_group_words(v, andword);
    if (i > 0) w += std::string(" ") + kScale[i];
    parts.push_back(std::move(w));
  }
  if (parts.empty()) return std::string(zero);
  std::string out;
  for (size_t k = 0; k < parts.size(); ++k) {
    if (k > 0) {
      bool base_part = (k == parts.size() - 1) && (groups[0] > 0);
      if (base_part && groups[0] < 100)
        out += (andword && andword[0]) ? std::string(" ") + andword + " " : " ";
      else
        out += ", ";
    }
    out += parts[k];
  }
  return out;
}

// normalizer.py L124-143 (_expand_number: group=2 "oh" two-digit pairs)
std::string num_words_group2(uint64_t n, const char* zero) {
  std::string d = std::to_string(n);
  if (d.size() % 2) d = "0" + d;
  std::string out;
  for (size_t i = 0; i + 1 < d.size(); i += 2) {
    int v = (d[i] - '0') * 10 + (d[i + 1] - '0');
    std::string w;
    if (v == 0) w = std::string(zero) + " " + zero;
    else if (v < 10) w = std::string(zero) + " " + kUnit[v];
    else if (v < 20) w = kTeen[v - 10];
    else {
      int t = v / 10, o = v % 10;
      w = kTen[t];
      if (o) { w += "-"; w += kUnit[o]; }
    }
    if (!out.empty()) out += " ";
    out += w;
  }
  return out;
}

// normalizer.py L121-123 (_expand_ordinal: inflect ordinal substitution)
std::string sub_ord(const std::string& s) {
  static const char* kBase[] = {"one", "two", "three", "five", "eight", "nine",
                                "twelve"};
  static const char* kOrd[] = {"first", "second", "third", "fifth", "eighth",
                               "ninth", "twelfth"};
  for (int i = 0; i < 7; i++) {
    size_t blen = std::strlen(kBase[i]);
    if (s.size() >= blen && s.compare(s.size() - blen, blen, kBase[i]) == 0)
      return s.substr(0, s.size() - blen) + kOrd[i];
  }
  if (s.size() >= 2 && s.compare(s.size() - 2, 2, "ty") == 0)
    return s.substr(0, s.size() - 2) + "tieth";
  return s + "th";
}

// normalizer.py L124-143 (_expand_number defaults: andword="and", zero="zero")
std::string number_to_words_plain(uint64_t n) { return num_words_group0(n, "and", "zero"); }

// normalizer.py L121-123 (_expand_ordinal: strip suffix, read number, ordinal-substitute)
std::string number_to_words_suffix(const std::string& s) {
  if (s.size() < 3) return s;
  uint64_t n = 0;
  for (size_t i = 0; i + 2 < s.size(); i++)
    if (isdigit8((unsigned char)s[i])) n = n * 10 + (uint64_t)(s[i] - '0');
  return sub_ord(num_words_group0(n, "and", "zero"));
}

// ---------------------------------------------------------------------------
// zipvoice/tokenizer/normalizer.py L17-160 (EnglishTextNormalizer)
// ---------------------------------------------------------------------------
// normalizer.py L155-159 (expand_abbreviations word-boundary, case-insensitive replacement)
std::string replace_word_ci(const std::string& text, const char* pat, const char* rep) {
  size_t n = std::strlen(pat);
  std::string out;
  size_t last = 0;
  for (size_t i = 0; i + n <= text.size(); i++) {
    bool match = true;
    for (size_t k = 0; k < n; k++) {
      if (std::tolower((unsigned char)text[i + k]) != pat[k]) { match = false; break; }
    }
    if (!match) continue;
    size_t pre = i, post = i + n;
    bool bpre = (pre == 0) || !is_word_char((unsigned char)text[pre - 1]);
    bool bpost = (post == text.size()) || !is_word_char((unsigned char)text[post]);
    if (bpre && bpost) {
      out.append(text, last, i - last);
      out += rep;
      last = post;
      i = post - 1;
    }
  }
  out.append(text, last, text.size() - last);
  return out;
}

std::string expand_abbreviations(std::string text) {
// normalizer.py L155-159 (expand_abbreviations table)
  static const std::pair<const char*, const char*> kAbbrev[] = {
      {"mrs", "misess"},   {"mr", "mister"},   {"dr", "doctor"},  {"st", "saint"},
      {"co", "company"},   {"jr", "junior"},   {"maj", "major"},  {"gen", "general"},
      {"drs", "doctors"},  {"rev", "reverend"}, {"lt", "lieutenant"}, {"hon", "honorable"},
      {"sgt", "sergeant"}, {"capt", "captain"}, {"esq", "esquire"}, {"ltd", "limited"},
      {"col", "colonel"},  {"ft", "fort"},     {"etc", "et cetera"}, {"btw", "by the way"},
  };
  for (const auto& kv : kAbbrev) text = replace_word_ci(text, kv.first, kv.second);
  return text;
}

// normalizer.py L145 (normalize_numbers) + L87-89 (_remove_commas)
std::string pass_comma_numbers(const std::string& s) {
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    if (isdigit8((unsigned char)s[i])) {
      size_t j = i + 1;
      while (j < s.size() && (isdigit8((unsigned char)s[j]) || s[j] == ',')) j++;
      size_t end = j - 1;
      while (end > i && s[end] == ',') end--;
      if (end - i >= 2) {
        for (size_t k = i; k <= end; k++)
          if (s[k] != ',') out += s[k];
        i = end + 1;
        continue;
      }
    }
    out += s[i];
    i++;
  }
  return out;
}

// Pass 2: pounds re £([0-9\,]*[0-9]+) -> "<group> pounds" (digits stay digits;
// normalizer.py L146 (normalize_numbers, _pounds_re)
std::string pass_pounds(const std::string& s) {
  static const char kPound[] = "\xc2\xa3";
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    if (s.compare(i, 2, kPound) == 0) {
      size_t j = i + 2;
      while (j < s.size() && (isdigit8((unsigned char)s[j]) || s[j] == ',')) j++;
      size_t end = j - 1;
      while (end > i + 1 && s[end] == ',') end--;
      if (end >= i + 2 && isdigit8((unsigned char)s[end])) {
        out.append(s, i + 2, end - (i + 2) + 1);
        out += " pounds";
        i = end + 1;
        continue;
      }
    }
    out += s[i];
    i++;
  }
  return out;
}

// normalizer.py L145-152 (digit-run parsing for the number passes)
uint64_t parse_digits(const char* p, size_t n) {
  uint64_t v = 0;
  for (size_t k = 0; k < n; k++)
    if (isdigit8((unsigned char)p[k])) v = v * 10 + (uint64_t)(p[k] - '0');
  return v;
}

// normalizer.py L70-86 (fraction_to_words)
std::string fraction_to_words(uint64_t num, uint64_t den) {
  if (num == 1 && den == 2) return " one half ";
  if (num == 1 && den == 4) return " one quarter ";
  if (den == 2) return " " + number_to_words_plain(num) + " halves ";
  if (den == 4) return " " + number_to_words_plain(num) + " quarters ";
  return " " + number_to_words_plain(num) + " " +
         sub_ord(num_words_group0(den, "and", "zero")) + " ";
}

// normalizer.py L147 (normalize_numbers) + L90-109 (_expand_dollars)
std::string pass_dollars(const std::string& s) {
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    if (s[i] == '$') {
      size_t j = i + 1;
      while (j < s.size() && (isdigit8((unsigned char)s[j]) || s[j] == '.' || s[j] == ',')) j++;
      size_t end = j - 1;
      while (end > i && s[end] == ',') end--;
      if (end >= i + 1 && isdigit8((unsigned char)s[end])) {
        std::string group = s.substr(i + 1, end - i);
        // normalizer.py L90-109 (_expand_dollars: parts = match.split("."))
        size_t d1 = group.find('.');
        std::string dollars_str = d1 == std::string::npos ? group : group.substr(0, d1);
        std::string cents_str;
        bool many = false;
        if (d1 != std::string::npos) {
          size_t d2 = group.find('.', d1 + 1);
          if (d2 != std::string::npos) many = true;
          cents_str = group.substr(d1 + 1);
        }
        if (many) {
          out += " ";
          out += group;
          out += " dollars ";
        } else {
          uint64_t dollars = dollars_str.empty() ? 0 : parse_digits(dollars_str.data(), dollars_str.size());
          uint64_t cents = cents_str.empty() ? 0 : parse_digits(cents_str.data(), cents_str.size());
          if (dollars && cents) {
            // Python: " %s %s, %s %s " % (dollars, unit, cents, unit) — the
            // normalizer.py L90-109 (_expand_dollars format; digits -> words in the number pass)
            out += " ";
            out += std::to_string(dollars);
            out += (dollars == 1) ? " dollar" : " dollars";
            out += ", ";
            out += std::to_string(cents);
            out += (cents == 1) ? " cent" : " cents";
            out += " ";
          } else if (dollars) {
            out += " ";
            out += std::to_string(dollars);
            out += (dollars == 1) ? " dollar " : " dollars ";
          } else if (cents) {
            out += " ";
            out += std::to_string(cents);
            out += (cents == 1) ? " cent " : " cents ";
          } else {
            out += " zero dollars ";
          }
        }
        i = end + 1;
        continue;
      }
    }
    out += s[i];
    i++;
  }
  return out;
}

// normalizer.py L148 (normalize_numbers) + L110-114 (_expand_fraction)
std::string pass_fractions(const std::string& s) {
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    if (isdigit8((unsigned char)s[i])) {
      size_t a = i;
      while (a < s.size() && isdigit8((unsigned char)s[a])) a++;
      if (a < s.size() && s[a] == '/') {
        size_t b = a + 1;
        while (b < s.size() && isdigit8((unsigned char)s[b])) b++;
        if (b > a + 1) {
          out += fraction_to_words(parse_digits(s.data() + i, a - i), parse_digits(s.data() + a + 1, b - a - 1));
          i = b;
          continue;
        }
      }
    }
    out += s[i];
    i++;
  }
  return out;
}

// normalizer.py L149 (normalize_numbers) + L115-117 (_expand_decimal_point)
std::string pass_decimals(const std::string& s) {
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    if (isdigit8((unsigned char)s[i])) {
      size_t a = i;
      while (a < s.size() && isdigit8((unsigned char)s[a])) a++;
      if (a < s.size() && s[a] == '.') {
        size_t b = a + 1;
        while (b < s.size() && isdigit8((unsigned char)s[b])) b++;
        if (b > a + 1) {
          out.append(s, i, a - i);
          out += " point ";
          out.append(s, a + 1, b - a - 1);
          i = b;
          continue;
        }
      }
    }
    out += s[i];
    i++;
  }
  return out;
}

// normalizer.py L150 (normalize_numbers) + L118-120 (_expand_percent)
std::string pass_percent(const std::string& s) {
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    if (isdigit8((unsigned char)s[i])) {
      size_t j = i;
      while (j < s.size() &&
             (isdigit8((unsigned char)s[j]) || s[j] == '.' || s[j] == ','))
        j++;
      if (j < s.size() && s[j] == '%' && isdigit8((unsigned char)s[j - 1])) {
        out.append(s, i, j - i);
        out += " percent ";
        i = j + 1;
        continue;
      }
      out.append(s, i, j - i);
      i = j;
      continue;
    }
    out += s[i];
    i++;
  }
  return out;
}

// normalizer.py L151 (normalize_numbers) + L121-123 (_expand_ordinal)
std::string pass_ordinals(const std::string& s) {
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    if (isdigit8((unsigned char)s[i])) {
      size_t a = i;
      while (a < s.size() && isdigit8((unsigned char)s[a])) a++;
      if (a + 2 <= s.size()) {
        const char* suf = s.data() + a;
        if ((suf[0] == 's' && suf[1] == 't') ||
            (suf[0] == 'n' && suf[1] == 'd') ||
            (suf[0] == 'r' && suf[1] == 'd') ||
            (suf[0] == 't' && suf[1] == 'h')) {
          out += " ";
          out += number_to_words_suffix(std::string(s.data() + i, (a - i) + 2));
          out += " ";
          i = a + 2;
          continue;
        }
      }
    }
    out += s[i];
    i++;
  }
  return out;
}

// normalizer.py L152 (normalize_numbers, _number_re)
std::string expand_number(uint64_t num) {
  if (num > 1000 && num < 3000) {
    if (num == 2000) return " two thousand ";
    if (num > 2000 && num < 2010)
      return " two thousand " + number_to_words_plain(num % 100) + " ";
    if (num % 100 == 0) return " " + number_to_words_plain(num / 100) + " hundred ";
    return " " + num_words_group2(num, "oh") + " ";
  }
  return " " + num_words_group0(num, "", "zero") + " ";
}

// normalizer.py L145-152 (normalize_numbers: scan digit-runs, expand each via expand_number)
std::string pass_numbers(const std::string& s) {
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    if (isdigit8((unsigned char)s[i])) {
      size_t a = i;
      while (a < s.size() && isdigit8((unsigned char)s[a])) a++;
      out += expand_number(parse_digits(s.data() + i, a - i));
      i = a;
      continue;
    }
    out += s[i];
    i++;
  }
  return out;
}

// ---------------------------------------------------------------------------
// zipvoice/tokenizer/tokenizer.py L158-166 (EspeakTokenizer.g2p, piper_phonemize)
// ---------------------------------------------------------------------------

// tokenizer.py L158-166 (EspeakTokenizer.g2p: NFD-decompose each clause)
void nfd_push(std::vector<std::string>& out, uint32_t cp) {
  static const struct {
    uint32_t pre, post[2];
  } kDecomp[] = {
      {0x00E0, {'a', 0x0300}}, {0x00E1, {'a', 0x0301}}, {0x00E3, {'a', 0x0303}},
      {0x00E4, {'a', 0x0308}}, {0x00E5, {'a', 0x030A}},
      {0x00E7, {'c', 0x0327}},
      {0x00E8, {'e', 0x0300}}, {0x00E9, {'e', 0x0301}}, {0x00EA, {'e', 0x0302}},
      {0x00EB, {'e', 0x0308}},
      {0x00EC, {'i', 0x0300}}, {0x00ED, {'i', 0x0301}}, {0x00EE, {'i', 0x0302}},
      {0x00EF, {'i', 0x0308}},
      {0x00F1, {'n', 0x0303}},
      {0x00F2, {'o', 0x0300}}, {0x00F3, {'o', 0x0301}}, {0x00F4, {'o', 0x0308}},
      {0x00F6, {'o', 0x0338}},
      {0x00F9, {'u', 0x0300}}, {0x00FA, {'u', 0x0301}}, {0x00FB, {'u', 0x0302}},
      {0x00FC, {'u', 0x0308}},
      {0x00FD, {'y', 0x0308}},
  };
  for (const auto& d : kDecomp)
    if (d.pre == cp) {
      out.push_back(utf8_encode(d.post[0]));
      out.push_back(utf8_encode(d.post[1]));
      return;
    }
  out.push_back(utf8_encode(cp));
}

void g2p_collect_clause(const char* clause, std::vector<std::string>& tokens) {
  bool in_lang = false;
  size_t i = 0;
  while (clause[i] != '\0') {
    size_t saved = i;
    uint32_t cp = utf8_next(clause, i);
    if (cp == 0xFFFD) {
      i = saved + 1;
      continue;
    }
    if (in_lang) {
      if (cp == ')') in_lang = false;
    } else if (cp == '(') {
      in_lang = true;
    } else {
      // tokenizer.py L158-166 (EspeakTokenizer.g2p: 'j' (U+006A) only, no U+02B2)
      if (cp == 0x02B2) continue;
      nfd_push(tokens, cp);
    }
  }
}

// tokenizer.py L158-166 (EspeakTokenizer.g2p: clause-terminator punctuation codes)
constexpr int kPunctPeriod = 40 | 0x00080000;
constexpr int kPunctQuestion = 40 | 0x2000 | 0x00080000;
constexpr int kPunctExclaim = 45 | 0x3000 | 0x00080000;
constexpr int kPunctComma = 20 | 0x1000 | 0x00040000;
constexpr int kPunctColon = 30 | 0x00040000;
constexpr int kPunctSemi = 30 | 0x1000 | 0x00040000;
constexpr int kClauseSentence = 0x00080000;

// tokenizer.py L158-166 (EspeakTokenizer.g2p espeak state)
struct EspeakG2P {
  EspeakG2P() {
    const char* data_dir = LUX_ESPEAK_DATA_DIR;
    int r = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_dir, 0);
    ok = (r >= 0);
    if (!ok) std::fprintf(stderr, "[lux] espeak-ng init failed (code %d)\n", r);
  }
  bool ok = false;
};

}

// normalizer.py L62-69 (EnglishTextNormalizer.normalize: expand_abbreviations + normalize_numbers)
std::string normalize_english_text(const std::string& text) {
  std::string t = expand_abbreviations(text);
  t = pass_comma_numbers(t);
  t = pass_pounds(t);
  t = pass_dollars(t);
  t = pass_fractions(t);
  t = pass_decimals(t);
  t = pass_percent(t);
  t = pass_ordinals(t);
  t = pass_numbers(t);
  return t;
}

// tokenizer.py L369-386 (EmiliaTokenizer.map_punctuations)
std::string map_punctuations(const std::string& text) {
  std::string t = text;
  t = string_replace_all(t, "\xef\xbc\x8c", ",");
  t = string_replace_all(t, "\xe3\x80\x82", ".");
  t = string_replace_all(t, "\xef\xbc\x81", "!");
  t = string_replace_all(t, "\xef\xbf\xbf", "?");
  t = string_replace_all(t, "\xef\xbf\x9b", ";");
  t = string_replace_all(t, "\xef\xbf\x9a", ":");
  t = string_replace_all(t, "\xe3\x80\x81", ",");
  t = string_replace_all(t, "\xe2\x80\x98", "'");
  t = string_replace_all(t, "\xe2\x80\x9c", "\"");
  t = string_replace_all(t, "\xe2\x80\x9d", "\"");
  t = string_replace_all(t, "\xe2\x80\x99", "'");
  t = string_replace_all(t, "\xe2\x8b\xaf", "\xe2\x80\xa6");
  t = string_replace_all(t, "\xc2\xb7\xc2\xb7\xc2\xb7", "\xe2\x80\xa6");
  t = string_replace_all(t, "\xe3\x83\xbb\xe3\x83\xbb\xe3\x83\xbb", "\xe2\x80\xa6");
  t = string_replace_all(t, "...", "\xe2\x80\xa6");
  return t;
}

// tokenizer.py L387-474 (EmiliaTokenizer.get_segment + split_segments)
std::vector<Segment> split_language_segments(const std::string& text) {
  if (text.empty()) return {};

  // tokenizer.py L412-413 (get_segment _part_pattern findall: each part is a character, or a <...>/[...] string)
  std::vector<std::string> parts;
  size_t i = 0;
  while (i < text.size()) {
    if (text[i] == '<' || text[i] == '[') {
      size_t j = i + 1;
      while (j < text.size() && text[j] != '>' && text[j] != ']') j++;
      if (j < text.size()) {
        parts.push_back(text.substr(i, j - i + 1));
        i = j + 1;
        continue;
      }
    }
    size_t len = utf8_codepoint_len(text, i);
    parts.push_back(text.substr(i, len));
    i += len;
  }

  // tokenizer.py L387-447 (get_segment per-part language: CJK -> zh, <pinyin> -> zh, single ASCII letter -> en, else other)
  std::vector<std::string> types;
  for (const auto& p : parts) {
    bool zh = false;
    bool single = false;
    {
      size_t k = 0;
      uint32_t cp = utf8_next(p, k);
      if (p.size() > 0 && k == p.size()) {
        single = true;
        if (cp >= 0x4E00 && cp <= 0x9FA5) zh = true;
      }
    }
    if (!zh && p.size() > 1 && p.front() == '<' && p.back() == '>') zh = true;
    if (zh)
      types.push_back("zh");
    else if (single && is_ascii_alpha(p[0]))
      types.push_back("en");
    else
      types.push_back("other");
  }

  // tokenizer.py L387-447 (get_segment merge: an "other" run adopts the next part's language)
  std::vector<Segment> segments;
  std::string temp_seg;
  std::string temp_lang;
  for (size_t idx = 0; idx < types.size(); idx++) {
    if (idx == 0) {
      temp_seg = parts[0];
      temp_lang = types[0];
      continue;
    }
    if (temp_lang == "other") {
      temp_seg += parts[idx];
      temp_lang = types[idx];
    } else if (types[idx] == temp_lang || types[idx] == "other") {
      temp_seg += parts[idx];
    } else {
      segments.push_back({temp_seg, temp_lang});
      temp_seg = parts[idx];
      temp_lang = types[idx];
    }
  }
  segments.push_back({temp_seg, temp_lang});

  // tokenizer.py L448-474 (split_segments: pull <pinyin> / [tag] parts out of each segment)
  std::vector<Segment> result;
  for (auto& seg : segments) {
    size_t k = 0;
    while (k < seg.text.size()) {
      size_t part_end = std::string::npos;
      if (seg.text[k] == '<' || seg.text[k] == '[') {
        size_t j = k + 1;
        while (j < seg.text.size() && seg.text[j] != '>' && seg.text[j] != ']') j++;
        if (j < seg.text.size()) part_end = j;
      }
      if (part_end != std::string::npos) {
        std::string part = seg.text.substr(k, part_end - k + 1);
        std::string lang = seg.lang;
        if (part.size() > 1 && part.front() == '<' && part.back() == '>') lang = "pinyin";
        if (part.size() > 1 && part.front() == '[' && part.back() == ']') lang = "tag";
        result.push_back({part, lang});
        k = part_end + 1;
      } else {
        size_t j = k + 1;
        while (j < seg.text.size() && seg.text[j] != '<' && seg.text[j] != '[') j++;
        result.push_back({seg.text.substr(k, j - k), seg.lang});
        k = j;
      }
    }
  }
  return result;
}

Tokenizer Tokenizer::load(const std::string& token_file, std::string* err) {
  // tokens.txt format: "<token>\t<id>" per line.
  std::ifstream f(token_file);
  if (!f) {
    std::string r = ("cannot open " + token_file);
    if (err) *err = r;
    return Tokenizer{};
  }
  std::string content((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  return load_str(content, err);
}

Tokenizer Tokenizer::load_str(const std::string& table, std::string* err) {
  // tokenizer.py L202-233 (EmiliaTokenizer.__init__: "<token>\t<id>" per line, rstrip().split("\t"))
  Tokenizer t;
  size_t pos = 0;
  while (pos < table.size()) {
    size_t nl = table.find('\n', pos);
    std::string line = table.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
    pos = (nl == std::string::npos) ? table.size() : nl + 1;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    auto tab = line.find('\t');
    if (tab == std::string::npos) continue;
    std::string tok = line.substr(0, tab);
    size_t tab2 = line.find('\t', tab + 1);
    std::string id_field = line.substr(tab + 1, tab2 == std::string::npos ? std::string::npos : tab2 - tab - 1);
    int id = 0;
    for (char c : id_field)
      if (isdigit8((unsigned char)c)) id = id * 10 + (c - '0');
    t.token2id.emplace(std::move(tok), id);
  }
  t.vocab_size = (int)t.token2id.size();
  t.pad_id = t.token2id.count("_") ? t.token2id.at("_") : 0;
  return t;
}

// tokenizer.py L234-297 (EmiliaTokenizer.texts_to_token_ids: preprocess_text -> texts_to_tokens -> tokens_to_token_ids)
std::vector<int> Tokenizer::texts_to_token_ids(const std::string& text) const {
  std::string pre = map_punctuations(text);
  std::vector<std::string> tokens;
  for (const auto& seg : split_language_segments(pre)) {
    if (seg.lang == "en") {
      // tokenizer.py L321-330 (tokenize_EN: normalize then g2p)
      std::string norm = normalize_english_text(seg.text);
      auto phones = g2p_espeak_en_us(norm);
      tokens.insert(tokens.end(), phones.begin(), phones.end());
    } else if (seg.lang == "tag") {
      // tokenizer.py L246-278 (texts_to_tokens: a tag is its own token, e.g. [S1]) — the tag is its own token (e.g. [S1]).
      tokens.push_back(seg.text);
    } else if (seg.lang == "zh" || seg.lang == "pinyin") {
      // tokenizer.py L298-320 (tokenize_ZH: jieba+pypinyin)
      std::fprintf(stderr,
                   "[lux] warning: skipping %s segment: %.*s\n",
                   seg.lang.c_str(), (int)std::min<size_t>(seg.text.size(), 80), seg.text.data());
    } else {
      // tokenizer.py L246-278 (texts_to_tokens "skipping segment of unknown language" path)
      std::fprintf(stderr,
                   "[lux] warning: no English or Chinese characters found, "
                   "skipping segment of unknown language: %.*s\n",
                   (int)std::min<size_t>(seg.text.size(), 80), seg.text.data());
    }
  }
  // tokenizer.py L279-297 (tokens_to_token_ids: OOV tokens skipped)
  std::vector<int> ids;
  ids.reserve(tokens.size());
  for (const auto& tok : tokens) {
    auto it = token2id.find(tok);
    if (it != token2id.end()) ids.push_back(it->second);
  }
  return ids;
}

// tokenizer.py L158-166 (EspeakTokenizer.g2p; L161 reduce(x + y) flattening)
std::vector<std::string> g2p_espeak_en_us(const std::string& text) {
// tokenizer.py L158-166 (EspeakTokenizer.g2p clause loop)
  static EspeakG2P espeak;
  std::vector<std::string> tokens;
  if (!espeak.ok) return tokens;
  espeak_SetVoiceByName("en-us");
  std::string copy = text;
  const char* rest = copy.c_str();
  while (rest != nullptr) {
    int terminator = 0;
    const char* clause = espeak_TextToPhonemesWithTerminator(
        reinterpret_cast<const void**>(&rest), espeakCHARS_AUTO, 0x02 /*IPA*/, &terminator);
    if (clause == nullptr) break;
    g2p_collect_clause(clause, tokens);
    int punct = terminator & 0x000FFFFF;
    if (punct == kPunctPeriod) {
      tokens.push_back(".");
    } else if (punct == kPunctQuestion) {
      tokens.push_back("?");
    } else if (punct == kPunctExclaim) {
      tokens.push_back("!");
    } else if (punct == kPunctComma) {
      tokens.push_back(",");
      tokens.push_back(" ");
    } else if (punct == kPunctColon) {
      tokens.push_back(":");
      tokens.push_back(" ");
    } else if (punct == kPunctSemi) {
      tokens.push_back(";");
      tokens.push_back(" ");
    }
    (void)kClauseSentence;
  }
  return tokens;
}

}
