#include "duplex.h"
#include <cassert>
#include <chrono>
#include <iostream>

using namespace lora_omni;
using J = DuplexJson;
using namespace std::chrono_literals;
static J text(const char * value) { return J::array({{{"type", "text"}, {"text", value}}}); }
static DuplexSession::Event until(DuplexSession & session, const std::string & name, int seq = -1) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    DuplexSession::Event e;
    while (std::chrono::steady_clock::now() < deadline && session.next(e, [] { return false; })) {
        if (e.name == "error") throw std::runtime_error(e.data.dump());
        if (e.name == name && (seq < 0 || e.data.value("sequence", -1) == seq)) return e;
    }
    throw std::runtime_error("timeout waiting for " + name);
}
template<class F> static void rejects(int status, F f) {
    try { f(); assert(false); } catch (const DuplexError & e) { assert(e.status == status); }
}
int main() {
    std::atomic<bool> speaking{false}, overlap{false}, finish{false};
    std::mutex snapshots_mutex;
    std::vector<J> snapshots;
    DuplexSession session("test", J::array(), [&](const J & messages, bool, const auto &) {
        if (speaking) overlap = true;
        { std::lock_guard<std::mutex> lock(snapshots_mutex); snapshots.push_back(messages); }
        return DuplexReply{"answer", {}};
    }, [&](const auto &, const auto & stop, const auto & audio, const auto &) {
        speaking = true;
        audio(Vector(480, .1f));
        while (!stop() && !finish) std::this_thread::sleep_for(1ms);
        speaking = false;
    }, {});
    session.start();
    session.append(0, text("first"), true);
    const auto first = until(session, "speech_started");
    until(session, "audio_chunk");
    rejects(409, [&] { session.append(0, text("duplicate"), true); });
    session.append(1, text("next fragment"), false);
    until(session, "input_processed", 1);
    assert(overlap);
    session.append(2, text("final fragment"), true);
    until(session, "input_processed", 2);
    rejects(429, [&] { session.append(3, text("too early"), true); });
    const auto generation = first.data.at("generation_id").get<std::string>();
    assert(!session.cancel("old-generation")["cancelled"].get<bool>());
    assert(session.cancel(generation)["cancelled"].get<bool>());
    assert(!session.cancel(generation)["cancelled"].get<bool>());
    // A ready reply based on the cancelled assistant must be recomputed.
    auto recomputed = until(session, "input_processed", 2);
    assert(recomputed.data.at("revision").get<int>() == 2);
    auto second = until(session, "speech_started");
    assert(second.data.at("generation_id") != generation);
    {
        std::lock_guard<std::mutex> lock(snapshots_mutex);
        const auto & last = snapshots.back();
        assert(last.size() == 2); // two user turns; no cancelled assistant
        assert(last[0]["role"] == "user" && last[1]["content"].size() == 1);
        assert(last[1]["content"][0]["text"] == "next fragmentfinal fragment");
    }
    finish = true;
    auto done = until(session, "speech_done");
    session.acknowledge(second.data.at("generation_id").get<std::string>(), done.data.at("samples").get<uint64_t>());
    assert(!session.cancel(second.data.at("generation_id").get<std::string>())["cancelled"].get<bool>());
    session.close();
    rejects(410, [&] { session.append(3, text("closed"), true); });

    // Closing a stalled output reader must wake both workers, including emit.
    DuplexSession::Limits limits;
    limits.events = 1;
    limits.input_bytes = 80;
    DuplexSession stalled("stalled", J::array(), [](const auto &, bool, const auto &) {
        return DuplexReply{"answer", {}};
    }, [](const auto &, const auto &, const auto &, const auto &) {}, limits);
    stalled.start();
    rejects(413, [&] { stalled.append(0, text(std::string(100, 'x').c_str()), false); });
    rejects(400, [&] { stalled.append(0, J::array(), false); });
    stalled.append(0, text("ok"), true);
    stalled.close();

    DuplexSession failure("failure", J::array(), [](const auto &, bool, const auto &) -> DuplexReply {
        throw std::runtime_error("injected failure");
    }, [](const auto &, const auto &, const auto &, const auto &) {}, {});
    failure.start(); failure.append(0, text("fail"), true);
    DuplexSession::Event error;
    while (failure.next(error, [] { return false; }) && error.name != "error") {}
    assert(error.name == "error" && error.data["error"] == "injected failure");
    failure.close();
    J merged_audio;
    DuplexSession audio("audio", J::array(), [&](const J & messages, bool final, const auto &) {
        if (final) merged_audio = messages.back()["content"];
        return DuplexReply{"ok", {}};
    }, [](const auto &, const auto &, const auto &, const auto &) {}, {});
    auto pcm = [](const char * data, int rate = 16000) {
        return J::array({{{"type", "input_audio"}, {"input_audio", {{"format", "pcm16"},
            {"sample_rate", rate}, {"channels", 1}, {"data", data}}}}});
    };
    audio.start();
    rejects(400, [&] { audio.append(0, pcm("AA=="), false); });
    audio.append(0, pcm("AAABAA=="), false);
    until(audio, "input_processed", 0);
    rejects(400, [&] { audio.append(1, pcm("AgADAA==", 24000), true); });
    audio.append(1, pcm("AgADAA=="), true);
    until(audio, "speech_done");
    audio.close();
    assert(merged_audio.size() == 1);
    assert(merged_audio[0]["input_audio"]["format"] == "wav");
    assert(merged_audio[0]["input_audio"]["data"] == "UklGRiwAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQgAAAAAAAEAAgADAA==");
    J heard_history;
    DuplexSession played("played", J::array(), [&](const J & messages, bool, const auto &) {
        heard_history = messages;
        return DuplexReply{"heard. unheard.", {}};
    }, [](const auto &, const auto &, const auto & audio, const auto & boundary) {
        audio(Vector(960, .1f)); boundary(7, 480); boundary(15, 960);
    }, {});
    played.start(); played.append(0, text("first"), true);
    auto generated = until(played, "speech_done");
    const auto gen = generated.data["generation_id"].get<std::string>();
    rejects(409, [&] { played.acknowledge(gen, 961); });
    assert(!played.acknowledge(gen, 0, false)["completed"].get<bool>());
    assert(!played.acknowledge(gen, 480)["completed"].get<bool>());
    rejects(409, [&] { played.acknowledge(gen, 479); });
    played.append(1, text("next"), true);
    until(played, "input_processed", 1);
    // Synthesis has ended but playback has not: cancellation still works.
    assert(played.cancel(gen)["cancelled"].get<bool>());
    until(played, "input_processed", 1);
    until(played, "speech_started"); played.close();
    assert(heard_history.size() == 3 && heard_history[1]["content"] == "heard. ");
    std::string joined;
    DuplexSession tiny("tiny", J::array(), [&](const J & messages, bool final, const auto &) {
        if (final) joined = messages.back()["content"][0]["text"].template get<std::string>();
        return DuplexReply{"ok", {}};
    }, [](const auto &, const auto &, const auto &, const auto &) {}, {});
    tiny.start();
    for (int i = 0; i < 100; ++i) tiny.append(i, text("x"), i == 99);
    until(tiny, "speech_done"); tiny.close();
    assert(joined == std::string(100, 'x'));
    // Auxiliary queries (ASR/visual/web planning) use the same Thinker worker,
    // while speech proceeds on its independent worker. Replacement snapshots
    // must not duplicate the VAD pre-roll or retain an older screen frame.
    std::thread::id thinker_id;
    std::atomic<bool> talking{false};
    J replaced;
    DuplexSession auxiliary("aux", J::array(), [&](const J & messages, bool, const auto &) {
        thinker_id = std::this_thread::get_id(); replaced = messages.back();
        return DuplexReply{"heard. unheard.", {}};
    }, [&](const auto &, const auto & stop, const auto & audio, const auto & boundary) {
        talking = true; audio(Vector(960, .1f)); boundary(7, 480);
        while (!stop()) std::this_thread::sleep_for(1ms);
        talking = false;
    }, {}, [&](const J &, const auto &) {
        assert(std::this_thread::get_id() == thinker_id && talking);
        return J{{"text", "auxiliary"}};
    });
    auxiliary.start(); auxiliary.append(0, text("old"), false);
    until(auxiliary, "input_processed", 0);
    auxiliary.append(1, text("replacement"), true, true, {{"frame_id", std::string(32, 'a')}});
    const auto started = until(auxiliary, "speech_started"); until(auxiliary, "audio_chunk");
    assert(replaced["content"][0]["text"] == "replacement");
    assert(replaced["_omni_visual"]["frame_id"] == std::string(32, 'a'));
    assert(auxiliary.query(J::object(), [] { return false; })["text"] == "auxiliary");
    until(auxiliary, "phrase_boundary");
    auto aid = started.data["generation_id"].get<std::string>();
    auxiliary.acknowledge(aid, 600);
    auto rolled_back = auxiliary.cancel(aid);
    assert(rolled_back["heard_text"] == "heard. " && rolled_back["heard_samples"] == 480 && rolled_back["played_samples"] == 600);
    auxiliary.close();
    // An early interruption can append another user question while the old
    // assistant is still active. Rollback must edit that assistant's slot.
    J observed;
    DuplexSession early("early", J::array(), [&](const J & messages, bool, const auto &) {
        observed = messages; return DuplexReply{"heard. unheard.", {}};
    }, [](const auto &, const auto &, const auto & audio, const auto & boundary) {
        audio(Vector(960, .1f)); boundary(7, 480); boundary(15, 960);
    }, {});
    early.start(); early.append(0, text("question"), true);
    auto edone = until(early, "speech_done"); auto egen = edone.data["generation_id"].get<std::string>();
    early.append(1, text("misfire"), false); until(early, "input_processed", 1);
    early.interrupt_input(1, J::array());
    rejects(409, [&] { early.interrupt_input(1, J::array()); });
    early.interrupt_input(-1, text("preserved early question"));
    early.acknowledge(egen, 480); early.cancel(egen);
    early.append(2, text("clarification"), true); until(early, "speech_done"); early.close();
    assert(observed.size() == 4 && observed[1]["content"] == "heard. ");
    assert(observed[2]["content"][0]["text"] == "preserved early question");
    // Compaction uses a revision CAS and preserves the active tail exactly.
    J initial = J::array({{{"role","system"},{"content","rules"}}});
    for (int i=0;i<6;++i) initial.push_back({{"role", i%2 ? "assistant" : "user"}, {"content", std::to_string(i)}});
    DuplexSession rebase("rebase", initial, [](const auto &, bool, const auto &) {
        return DuplexReply{"heard. unheard.", {}};
    }, [](const auto &, const auto &, const auto & audio, const auto & boundary) {
        audio(Vector(960,.1f)); boundary(7,480);
    }, {});
    rebase.start(); rebase.append(0,text("current"),true); auto rd = until(rebase,"speech_done");
    auto snap = rebase.context(J::object()); J compacted = J::array({initial[0]});
    auto h = snap["messages"]; for(size_t i=h.size()-4;i<h.size();++i) compacted.push_back(h[i]);
    J wrong=compacted; wrong.back()["content"]="altered active speech";
    rejects(409,[&]{rebase.context({{"revision",snap["revision"]},{"messages",wrong}});});
    rebase.context({{"revision",snap["revision"]},{"messages",compacted}});
    rejects(409,[&]{rebase.context({{"revision",snap["revision"]},{"messages",compacted}});});
    auto rgen=rd.data["generation_id"].get<std::string>();
    assert(!rebase.acknowledge(rgen,960,false)["completed"].get<bool>());
    rebase.cancel(rgen);
    assert(rebase.context(J::object())["messages"].back()["content"]=="heard. "); rebase.close();
    // Retrieval reaches Thinker in the system context, without becoming user
    // speech conditioning or growing the committed conversation each turn.
    for (const J & bootstrap : {J::array(), J::array({{{"role","system"},{"content","rules"}}}),
            J::array({{{"role","system"},{"content",text("rules")}},
                      {{"role","user"},{"content","old question"}},
                      {{"role","assistant"},{"content","old answer"}}})}) {
        J seen;
        DuplexSession grounding("grounding", bootstrap, [&](const J & messages, bool, const auto &) {
            seen=messages; return DuplexReply{"ok", {}};
        }, [](const auto &, const auto &, const auto & audio, const auto &) {audio(Vector(480,.1f));}, {});
        grounding.start();
        grounding.append(0,text("camera question"),true,true,J::object(),"camera question",32,"retrieved facts");
        auto gd=until(grounding,"speech_done");
        assert(seen.size()==bootstrap.size()+2);
        for (size_t i=0; i<bootstrap.size(); ++i) assert(seen[i]==bootstrap[i]);
        assert(seen[seen.size()-2]==J({{"role","system"},{"content","retrieved facts"}}));
        assert(seen.back()["content"]==text("camera question"));
        assert(grounding.context(J::object())["messages"].dump().find("retrieved facts")==std::string::npos);
        grounding.acknowledge(gd.data["generation_id"].get<std::string>(),480);
        grounding.append(1,text("next"),true);until(grounding,"speech_done");
        assert(seen.dump().find("retrieved facts")==std::string::npos);
        grounding.close();
    }
    // A live microphone appends faster than inference. Each immutable preview
    // still completes; only finalization or replacement supersedes its work.
    std::atomic<bool> preview_entered{false}, release_preview{false};
    DuplexSession growing("growing", J::array(), [&](const J &, bool, const auto & stop) {
        preview_entered=true;
        while (!release_preview && !stop()) std::this_thread::sleep_for(1ms);
        assert(!stop()); return DuplexReply{"preview", {}};
    }, [](const auto &, const auto &, const auto &, const auto &) {}, {});
    growing.start(); growing.append(0,text("first"),false);
    while (!preview_entered) std::this_thread::sleep_for(1ms);
    growing.append(1,text("second"),false); release_preview=true;
    until(growing,"input_processed",0); until(growing,"input_processed",1); growing.close();
    // Bound speech lead without blocking cancellation, and allow visual
    // continuation without inventing a second user utterance in history.
    DuplexSession::Limits paced_limits; paced_limits.playback_window=480; paced_limits.committed_history=true;
    std::atomic<int> emitted{0};
    DuplexSession paced("paced", J::array({{{"role","user"},{"content","original question"}}}),
        [](const auto &, bool, const auto &) {return DuplexReply{"one. two.",{}};},
        [&](const auto &, const auto &, const auto & audio, const auto & boundary) {
            for(int i=0;i<8;i++){audio(Vector(480,.1f));++emitted;if(i==0)boundary(4,480);}
        },paced_limits);
    paced.start(); paced.append(0,text("updated image"),true,true,J::object(),"",32,"",false);
    auto ps=until(paced,"speech_started"); until(paced,"audio_chunk",1);
    std::this_thread::sleep_for(30ms); assert(emitted==2);
    auto ph=paced.context(J::object())["messages"];
    assert(ph.size()==2 && ph[0]["content"]=="original question" && ph[1]["role"]=="assistant" && ph[1]["content"]=="");
    auto pg=ps.data["generation_id"].get<std::string>();
    paced.acknowledge(pg,480,false); until(paced,"audio_chunk",2);
    auto committed=paced.context(J::object())["messages"];
    assert(committed[1]["content"]=="one.");
    assert(paced.cancel(pg)["heard_text"]=="one.");
    assert(paced.context(J::object())["messages"]==committed); paced.close();
    J spoken_input;
    DuplexSession spoken("spoken", J::array(), [&](const J & messages, bool, const auto &) {
        spoken_input=messages.back();return DuplexReply{"Checking now.",{}};
    }, [](const auto &, const auto &, const auto & audio, const auto &) {audio(Vector(480,.1f));}, {});
    spoken.start();
    rejects(400,[&]{spoken.append(0,text("task"),false,true,J::object(),"",32,"",false,"Checking now.");});
    rejects(400,[&]{spoken.append(0,text("task"),true,true,J::object(),"",32,"",true,"Checking now.");});
    rejects(400,[&]{spoken.append(0,text("task"),true,true,J::object(),"",32,"",false,std::string(1025,'x'));});
    spoken.append(0,text("task"),true,true,J::object(),"",32,"",false,"Checking now.");
    until(spoken,"speech_done");assert(spoken_input["_omni_spoken_text"]=="Checking now.");
    assert(spoken.context(J::object())["messages"].size()==1);spoken.close();
    std::cout << "duplex overlap, revision rollback, ordering, limits, cancellation, disconnect and grounding: passed\n";
}
