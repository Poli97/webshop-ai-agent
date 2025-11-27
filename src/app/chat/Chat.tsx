import { SparklesIcon, XMarkIcon } from "@heroicons/react/24/outline";
import {
  AutoModel,
  AutoTokenizer,
  Message,
  PreTrainedModel,
  PreTrainedTokenizer,
} from "@huggingface/transformers";
import { type FC, type ReactElement, useRef, useState } from "react";

import usePageContext from "../../store/provider/pageContext/usePageContext.ts";
import { Loader } from "../../theme";
import cn from "../../utils/classnames.ts";
import mdToHtml from "../../utils/converter/mdToHtml.ts";
import { MODELS, SYSTEM_PROMPT } from "../../utils/llm/constants.ts";
import {
  WebMCPTool,
  webMCPToolToChatTemplateTool,
} from "../../utils/llm/webMcp.ts";
import ChatForm from "./ChatForm.tsx";

const Chat: FC = () => {
  const [chatOpen, setChatOpen] = useState<boolean>(false);

  const [thinking, setThinking] = useState<boolean>(false);
  const [response, setResponse] = useState<string>("");
  const [callbackElements, setCallbackElements] = useState<Array<ReactElement>>(
    []
  );

  const [conversation, setConversation] = useState<Array<Message>>([]);

  const tools: Array<WebMCPTool> = [
    {
      name: "get_page_context",
      description:
        "Get the current page context. Often the user navigates through the page so use this tool each time the user requests information about the current page or item",
      inputSchema: {
        type: "object",
        properties: {},
        required: [],
      },
      execute: async () => {
        return `Current Page: ${pageContext.title}
                  ${pageContext.content}`;
      },
    },
  ];

  const pipe = useRef<{
    tokenizer: PreTrainedTokenizer;
    model: PreTrainedModel;
  }>(null);

  const { pageContext } = usePageContext();

  const onAskLLM = async (question: string): Promise<string> => {
    const model = MODELS.granite3B;

    const messages = conversation;

    if (messages.length == 0) {
      messages.push({
        role: "system",
        content: SYSTEM_PROMPT,
      });
    }

    messages.push({
      role: "user",
      content: question,
    });

    if (!pipe.current) {
      const tokenizer = await AutoTokenizer.from_pretrained(model.modelId);
      const pretrainedModel: PreTrainedModel = await AutoModel.from_pretrained(
        model.modelId,
        {
          device: model.device,
          dtype: model.dtype,
        }
      );

      pipe.current = {
        tokenizer,
        model: pretrainedModel,
      };
    }

    const input: any = pipe.current.tokenizer.apply_chat_template(messages, {
      tools: tools.map(webMCPToolToChatTemplateTool),
      add_generation_prompt: true,
      return_dict: true,
    });

    const { sequences }: any = await pipe.current.model.generate({
      ...input,
      max_new_tokens: 1000,
      return_dict_in_generate: true,
    });

    console.log(messages);

    const lengthOfInput: number = input.input_ids_dims[1];
    const response = pipe.current.tokenizer.batch_decode(
      sequences.slice(null, [lengthOfInput, Number.MAX_SAFE_INTEGER], {
        skip_special_tokens: true,
      })
    )[0];

    console.log(response);

    messages.push({ role: "assistant", content: response });

    setConversation(messages);

    return response;
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
