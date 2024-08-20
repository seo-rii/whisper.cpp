#include "whisper.h"

#include <emscripten.h>
#include <emscripten/bind.h>

#include <vector>
#include <thread>

std::thread g_worker;

std::vector<struct whisper_context *> g_contexts(4, nullptr);

static inline int mpow2(int n) {
    int p = 1;
    while (p <= n) p *= 2;
    return p/2;
}

static std::string to_timestamp(int64_t t, bool comma = false) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);

    return std::string(buf);
}

EMSCRIPTEN_BINDINGS(whisper) {
    emscripten::function("init", emscripten::optional_override([](const std::string & path_model) {
        if (g_worker.joinable()) {
            g_worker.join();
        }

        for (size_t i = 0; i < g_contexts.size(); ++i) {
            auto param = whisper_context_default_params();
            param.dtw_token_timestamps = true;
            param.dtw_aheads_preset = WHISPER_AHEADS_BASE_EN;
            if (g_contexts[i] == nullptr) {
                g_contexts[i] = whisper_init_from_file_with_params(path_model.c_str(), param);
                if (g_contexts[i] != nullptr) {
                    return i + 1;
                } else {
                    return (size_t) 0;
                }
            }
        }

        return (size_t) 0;
    }));

    emscripten::function("free", emscripten::optional_override([](size_t index) {
        if (g_worker.joinable()) {
            g_worker.join();
        }

        --index;

        if (index < g_contexts.size()) {
            whisper_free(g_contexts[index]);
            g_contexts[index] = nullptr;
        }
    }));

    emscripten::function("full_default", emscripten::optional_override([](size_t index, const emscripten::val & audio, const std::string & lang, int nthreads, bool translate) {
        if (g_worker.joinable()) {
            g_worker.join();
        }

        --index;

        if (index >= g_contexts.size()) {
            return -1;
        }

        if (g_contexts[index] == nullptr) {
            return -2;
        }

        struct whisper_full_params params = whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);

        params.print_realtime   = false;
        params.print_progress   = false;
        params.print_timestamps = true;
        params.print_special    = false;
        params.translate        = translate;
        params.language         = whisper_is_multilingual(g_contexts[index]) ? lang.c_str() : "en";
        params.n_threads        = std::min(nthreads, std::min(16, mpow2(std::thread::hardware_concurrency())));
        params.offset_ms        = 0;
        params.token_timestamps = true;
        params.max_len          = 1;
        params.split_on_word    = true;

        std::vector<float> pcmf32;
        const int n = audio["length"].as<int>();

        emscripten::val heap = emscripten::val::module_property("HEAPU8");
        emscripten::val memory = heap["buffer"];

        pcmf32.resize(n);

        emscripten::val memoryView = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcmf32.data()), n);
        memoryView.call<void>("set", audio);


        // run the worker
        {
            g_worker = std::thread([index, params, pcmf32 = std::move(pcmf32)]() {
                whisper_reset_timings(g_contexts[index]);
                whisper_full(g_contexts[index], params, pcmf32.data(), pcmf32.size());
                std::string result;

                const int n_segments = whisper_full_n_segments(g_contexts[index]);
                float logprob_min = 0.0f;
                float logprob_sum = 0.0f;
                int   n_tokens    = 0;

                printf("!START\n");
                for (int i = 0; i < n_segments; ++i) {
                    auto t0 = whisper_full_get_segment_t0(g_contexts[index], i),
                         t1 = whisper_full_get_segment_t1(g_contexts[index], i);
                    const char * text = whisper_full_get_segment_text(g_contexts[index], i);

                    printf("[%s->%s] %s\n",
                            to_timestamp(t0).c_str(),
                            to_timestamp(t1).c_str(),
                            text);
                }
                printf("!DONE\n");
                fflush(stdout);
            });
        }

        return 0;
    }));
}
