import { SparklesIcon, XMarkIcon } from "@heroicons/react/24/outline";
import {
  Message,
  TextGenerationPipeline,
  pipeline,
} from "@huggingface/transformers";
import { type FC, type ReactElement, useRef, useState } from "react";

import { Loader } from "../../theme";
import cn from "../../utils/classnames.ts";
import mdToHtml from "../../utils/converter/mdToHtml.ts";
import { MODELS, SYSTEM_PROMPT } from "../../utils/llm/constants.ts";
import ChatForm from "./ChatForm.tsx";

const Chat: FC = () => {
  const [chatOpen, setChatOpen] = useState<boolean>(false);

  const [thinking, setThinking] = useState<boolean>(false);
  const [response, setResponse] = useState<string>("");
  const [callbackElements, setCallbackElements] = useState<Array<ReactElement>>(
    []
  );

  const pipe = useRef<TextGenerationPipeline | null>(null);

  const onAskLLM = async (question: string): Promise<string> => {
    const model = MODELS.granite1B;
    if (!pipe.current) {
      pipe.current = await pipeline("text-generation", model.modelId, {
        device: model.device,
        dtype: model.dtype,
        progress_callback: console.log,
      });
    }

    const messages: Array<Message> = [
      {
        role: "system",
        content: SYSTEM_PROMPT,
      },
      {
        role: "user",
        content: question,
      },
    ];

    //max_new_tokens should be larger then the number of tokens you expect in the response
    const resp = await pipe.current(messages, {
      max_new_tokens: 1024,
    });

    //@ts-expect-error transformer.js types issue
    return resp[0].generated_text.pop().content;
  };

  return (
    <>
      <div
        className={cn(
          "fixed right-4 bottom-24 flex w-md origin-bottom-right flex-col gap-4 rounded-lg border border-purple-400 bg-purple-50 p-6 shadow-xl transition duration-300",
          {
            "translate-x-0 translate-y-16 scale-15 opacity-0": !chatOpen,
          }
        )}
      >
        <h3 className="flex items-center gap-2">
          <SparklesIcon aria-hidden="true" className="size-4" /> Ask the Agent
        </h3>
        <ChatForm
          chatOpen={chatOpen}
          onSubmit={async (prompt) => {
            if (!prompt) {
              setResponse("");
              setCallbackElements([]);
              return;
            }

            setThinking(true);
            const resp = await onAskLLM(prompt);
            setResponse(resp);
            setThinking(false);
          }}
        />
        {(response.length !== 0 || thinking) && (
          <div className="mt-4">
            {thinking ? (
              <p className="flex items-center gap-3 font-light text-gray-500 italic">
                <Loader size={4} /> thinking..
              </p>
            ) : (
              <div className="flex flex-col gap-4">
                {callbackElements.map((element) => element)}
                <div
                  className="font-light text-gray-700 [&>li]:ml-5 [&>ol]:my-2 [&>ol]:ml-4 [&>ol]:list-decimal [&>ul]:my-2 [&>ul]:ml-5 [&>ul]:list-disc"
                  dangerouslySetInnerHTML={{ __html: mdToHtml(response) }}
                />
              </div>
            )}
          </div>
        )}
      </div>
      <button
        onClick={() => setChatOpen((open) => !open)}
        className="fixed right-4 bottom-4 grid cursor-pointer rounded-full bg-purple-900 p-3 text-white outline-2 outline-offset-4 outline-purple-300 transition hover:outline-4 hover:outline-purple-900 focus:outline-4 focus:outline-purple-900"
      >
        <XMarkIcon
          aria-hidden="true"
          className={cn("col-start-1 row-start-1 size-8 transition", {
            "rotate-90 opacity-0": !chatOpen,
          })}
        />
        <SparklesIcon
          aria-hidden="true"
          className={cn("col-start-1 row-start-1 size-8 transition", {
            "-rotate-90 opacity-0": chatOpen,
          })}
        />
      </button>
    </>
  );
};

export default Chat;
