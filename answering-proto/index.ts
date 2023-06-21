import { createInterface } from "readline/promises";
import { GithubRepoLoader } from "langchain/document_loaders/web/github";
import dotenv from "dotenv";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain } from "langchain/chains";
import { OpenAI } from "langchain/llms/openai";
dotenv.config();

const main = async () => {
  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  const repo =
    (await rl.question(
      "Enter the repo URL(Hit ENTER for the default repo): "
    )) || "https://github.com/open-sauced/ai";
  const branch =
    (await rl.question(
      "Enter the branch name(Hit ENTER for the default branch): "
    )) || "beta";

  console.log(`Using repo: ${repo} and branch: ${branch}`);
  console.log("Downloading repo, loading as Langchain docs, chunking...");
  console.time("Repo loading");

 //The request can be performed without a token, but unauthorized requests get timed out pretty quickly
 //You can get the token here: https://github.com/settings/tokens/new
 //No additional scopes are required
  const repoLoader = new GithubRepoLoader(repo, {
    branch,
    recursive: true,
    accessToken: process.env.GITHUB_ACCESS_TOKEN,
  });

  const docs = await repoLoader.loadAndSplit(
    new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    })
  );
  console.timeEnd("Repo loading");

  console.time("Embeddings generation");

  //The model used is text-embedding-ada-002
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  //I am using the memory store here, but as discussed we can use Supabase for persistence
  const store = await MemoryVectorStore.fromDocuments(docs, embeddings);
  console.timeEnd("Embeddings generation");

  const model = new OpenAI({
    temperature: 0,
    modelName: "gpt-3.5-turbo",
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  //The chunk size at L37 influences how many tokens are sent to GPT-3.5 when generating a response to the query. Larger the chunks, higher the costs
  const chain = RetrievalQAChain.fromLLM(model, store.asRetriever());

  while (true) {
    const query = await rl.question("Enter your query: ");
    const answer = await chain.call({
      query,
    });
    console.log(answer.text);
  }
};

main();
