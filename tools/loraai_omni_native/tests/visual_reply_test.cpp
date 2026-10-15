#include "visual_reply.h"
#include <cassert>
#include <iostream>
int main() {
    using json = common_json;
    for (size_t chunk = 1; chunk < 70; ++chunk) {
        int points = 0; std::string deltas;
        lora_omni::VisualReply result(std::string(32, 'a'), [&](const auto & kind, const json & value) {
            if (kind == "delta") deltas += value.at("text").get<std::string>();
            if (kind == "pointer") { ++points; assert(value.at("frame_id").get<std::string>() == std::string(32, 'a')); }
        });
        std::string input = "Нажмите здесь.<lora_pointer>{\"x\":0.4,\"y\":0.5,\"label\":\"Кнопка\"}</lora_pointer> Готово.";
        for (size_t i = 0; i < input.size(); i += chunk) result.push(input.substr(i, chunk));
        result.finish();
        assert(result.reply == "Нажмите здесь. Готово."); assert(deltas == result.reply); assert(points == 1);
    }
    for (auto command : {"{\"x\":-1,\"y\":0.5}", "{\"x\":0.9,\"y\":0.1,\"width\":0.4,\"height\":0.1}", "oops", "{\"x\":\"0.5\",\"y\":0.2}"}) {
        lora_omni::VisualReply result(std::string(32,'a'),{});
        result.push(std::string("Ответ.<lora_pointer>")+command+"</lora_pointer>"); result.finish();
        assert(result.pointers.empty()); assert(result.reply == "Ответ.");
    }
    lora_omni::VisualReply cancelled("",{});
    cancelled.push("Ответ.<lora_pointer>{\"x\":"); cancelled.finish();
    assert(cancelled.reply == "Ответ."); assert(cancelled.pointers.empty());
    for (bool allowed : {false, true}) {
        lora_omni::VisualReply action("", {}, allowed);
        const std::string input = "Show the screen.<lora_pointer>{\"request_screen\":true}</lora_pointer>";
        for (char c : input) action.push(std::string(1, c));
        action.finish();
        assert(action.reply == "Show the screen.");
        assert(action.pointers.empty());
        assert(action.actions.size() == (allowed ? 1 : 0));
    }
    std::cout << "PASS: UTF-8 token splits, pointer binding, speech filtering, malformed/truncated commands\n";
}
