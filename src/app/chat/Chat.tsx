import { SparklesIcon, XMarkIcon } from "@heroicons/react/24/outline";
import {
  AutoModel,
  AutoTokenizer,
  type Message,
  PreTrainedModel,
  PreTrainedTokenizer,
} from "@huggingface/transformers";
import { type FC, type ReactElement, useRef, useState } from "react";
import { useNavigate } from "react-router";

import { Category, Color, Size } from "../../store/products.ts";
import usePageContext from "../../store/provider/pageContext/usePageContext.ts";
import { Loader } from "../../theme";
import cn from "../../utils/classnames.ts";
import mdToHtml from "../../utils/converter/mdToHtml.ts";
import { MODELS, SYSTEM_PROMPT } from "../../utils/llm/constants.ts";
import {
  WebMCPTool,
  executeToolCall,
  extractToolCalls,
  webMCPToolToChatTemplateTool,
} from "../../utils/llm/webMcp.ts";
import findSimilarFAQs from "../../utils/vectorSearch/findSimilarFAQs.ts";
import ChatForm from "./ChatForm.tsx";

const Chat: FC = () => {
  const [chatOpen, setChatOpen] = useState<boolean>(false);

  const [conversation, setConversation] = useState<Array<Message>>([]);

  const [thinking, setThinking] = useState<boolean>(false);

  const [response, setResponse] = useState<string>("");

  const [callbackElements, setCallbackElements] = useState<Array<ReactElement>>(
    []
  );

  const navigate = useNavigate();

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
    {
      name: "open_product_overview",
      description: "Opens a product overview with a given set of filters",
      inputSchema: {
        type: "object",
        properties: {
          categories: {
            type: "string",
            description: `Can be one of the following values: ${Object.values(Category).join(", ")}`,
            default: "",
          },
          colors: {
            type: "string",
            description: `Can be one of the following values: ${Object.values(Color).join(", ")}`,
            default: "",
          },
          sizes: {
            type: "string",
            description: `Can be one of the following values: ${Object.values(Size).join(", ")}`,
            default: "",
          },
        },
        required: [],
      },
      execute: async (args: Record<string, any>) => {
        const query = Object.entries(args)
          .filter(([, value]) => value)
          .map(([key, value]) =>
            value
              .split(",")
              .map((value: string) => `${key}=${value.trim()}`)
              .join("&")
          )
          .join("&");

        navigate(`/products?${query}`);

        return `Tell the user you navigated to the product overview with "${query}"`;
      },
    },
    {
      name: "search_faqs",
      description:
        "IF the user ask a general question about the store, search fr the right answer in the FAQs",
      inputSchema: {
        type: "object",
        properties: {
          question: {
            type: "string",
            description: "The question to search for in the FAQs",
            default: "",
          },
        },
        required: ["question"],
      },
      execute: async (args: Record<string, any>) => {
        const { question } = args;

        const faqs = await findSimilarFAQs(question);
        return `Here are more information to answer the question:\n${faqs.join("\n")}`;
      },
    },
  ];

  const pipe = useRef<{
    tokenizer: PreTrainedTokenizer;

    model: PreTrainedModel;
  }>(null);

  const { pageContext } = usePageContext();

  console.log(pageContext);

  const runLLM = async (
    prompt: string,

    role: "tool" | "user" = "user"
  ): Promise<string> => {
    const model = MODELS.granite3B;

    const messages = conversation;

    if (messages.length == 0) {
      messages.push({
        role: "system",

        content: SYSTEM_PROMPT,
      });
    }

    messages.push({
      role,

      content: prompt,
    });

    if (!pipe.current) {
      const tokenizer = await AutoTokenizer.from_pretrained(model.modelId);

      const preTrainedModel = await AutoModel.from_pretrained(model.modelId, {
        device: model.device,

        dtype: model.dtype,
      });

      pipe.current = {
        tokenizer,

        model: preTrainedModel,
      };
    }

    console.log(messages);

    //const resp = await pipe.current(messages, { max_new_tokens: 1000 });

    const input = pipe.current.tokenizer.apply_chat_template(messages, {
      tools: tools.map(webMCPToolToChatTemplateTool),

      add_generation_prompt: true,

      return_dict: true,
    });

    const { sequences } = await pipe.current.model.generate({
      ...input,

      max_new_tokens: 1000,

      return_dict_in_generate: true,
    });

    const lengthOfInput: number = input.input_ids.dims[1];

    const response = pipe.current.tokenizer

      .batch_decode(
        sequences.slice(null, [lengthOfInput, Number.MAX_SAFE_INTEGER], {
          skip_special_tokens: true,
        })
      )[0]
      .replace("<|end_of_text|>", "");

    messages.push({
      role: "assistant",

      content: response,
    });

    setConversation(messages);

    return response;
  };

  const onAskLLM = async (question: string): Promise<string> => {
    let prompt = question;

    let role: "tool" | "user" = "user";

    const responses = [];

    while (prompt) {
      const response = await runLLM(prompt, role);

      const { toolCalls, message } = extractToolCalls(response);

      if (toolCalls.length === 0) {
        prompt = "";

        responses.push(message);
      } else {
        const toolResponses = await Promise.all(
          toolCalls.map(
            async (toolCall) => await executeToolCall(toolCall, tools)
          )
        );

        prompt = toolResponses.map((response) => response.result).join("\n\n");
      }

      role = "tool";
    }

    return responses.join("\n\n");
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
