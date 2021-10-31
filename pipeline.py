class Pipeline():
    def __init__(self):
       
      self.w2v_model=w2v_model

    def text_preprocessing(self,text):
        """
            The numbers, alphanumeric, punctuation and stopwords  will be removed from the text.
        """
        #words = set(nltk.corpus.words.words())
        #text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())
        sentences = sent_tokenize(text) #array of sentences
        sentences_clean = [ re.sub(r'[0-9]+[a-z]*','',sentence.lower()) for sentence in sentences] # remove number and alphanumeric
        sentences_clean = [ re.sub(r'[^\w\s]','',sentence) for sentence in sentences_clean] #remove punctuattion except whitespacce, tabs and newline
        
        stop_words = stopwords.words('english')
        sentence_tokens = [[ word for word in sentence.split(" ") if word not in stop_words and word] for sentence in sentences_clean]

        return sentence_tokens,sentences

    def Stemming(self,tokens):
        """
            Multiple stemmer Porter Stemmer (5 level suufix mapping rules) and Snowfall Stemmer
        """
        pass
    
    def pos(self, text):
        """
              This Function will do the part of the speech tagging.
        """
        doc = nlp(text)
        for token in doc:
          print(token, "->", token.pos_)
    
    def similar_words(self,word):
      """
          Find similar words of answer keywors
      """
      similar_w = self.w2v_model.most_similar(positive = word,topn=3)
      return [i[0] for i in similar_w]

    def top_sentences(self, text, sentences, num_ques=1):
        """
            Using text rank algorithm finding top sentences from the text
        """
        self.w2v=Word2Vec(text,size=1,min_count=1,iter=1000,window=2)
        sentence_embeddings=[[self.w2v[word][0] for word in words] for words in text]
        max_len=max([len(tokens) for tokens in text])
        sentence_embeddings=[np.pad(embedding,(0,max_len-len(embedding)),'constant') for embedding in sentence_embeddings]
        similarity_matrix = np.zeros([len(text), len(text)])
        for i,row_embedding in enumerate(sentence_embeddings):
          for j,column_embedding in enumerate(sentence_embeddings):
            similarity_matrix[i][j]=1-spatial.distance.cosine(row_embedding,column_embedding)
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        top_sentence={sentence:scores[index] for index,sentence in enumerate(sentences)}
        top=dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:num_ques])

        return top


    def keywords_extraction(self, sent_list):
        """
            Extract keywords from list of sentences using rake nltk
        """
        r = Rake()

        # Extraction given the list of strings where each string is a sentence.
        r.extract_keywords_from_text(sent_list)
        return r.get_ranked_phrases()
      
    def prediction(self,article):
        """
        An article/ a sentence passed as an input and fill up as an ouput 
        """
        text,sentences=self.text_preprocessing(article)

        top_sentences=self.top_sentences(text,sentences)
        itr=1;
        for key, value in top_sentences.items():
            main_keywords=self.keywords_extraction(key)
            #print(main_keywords)
            #genQuestion(key)
            print("Fill Up %s : %s  "%(str(itr),key.lower().replace(main_keywords[0],"______")))
            similar_options = w2v_model.most_similar(positive=main_keywords[0].split()[:2], topn = 3)
            #print(similar_options)
            similar_options=list(list(zip(*similar_options))[0])
            #print(similar_options)
            #similar_options = self.similar_words(main_keywords[0])
            similar_options.insert(3, main_keywords[0])
            shuffle(similar_options)
            print(similar_options)
            #print(shuffle(options))
            itr+=1
            print("\n")
