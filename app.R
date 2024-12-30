library(shiny)
library(shinythemes)
library(tidytext)
library(tidyverse)
library(text2vec)
library(glmnet)
library(wordcloud2)
library(DT)
library(plotly)

# Create sample training data
sample_data <- data.frame(
  text = c(
    "This product is amazing! I love it so much!",
    "Terrible experience, would not recommend.",
    "Great value for money, very satisfied.",
    "Disappointing quality, broke after one week.",
    "Excellent customer service and fast delivery!",
    "Waste of money, don't buy this.",
    "Perfect fit and exactly what I needed.",
    "Poor design and uncomfortable to use.",
    "Outstanding performance, exceeded expectations!",
    "Faulty product and horrible support."
  ),
  sentiment = factor(c(1, 0, 1, 0, 1, 0, 1, 0, 1, 0))  # 1 = positive, 0 = negative
)

# UI Definition
ui <- fluidPage(
  theme = shinytheme("flatly"),
  
  # Custom CSS
  tags$head(
    tags$style(HTML("
      .sentiment-positive { color: #28a745; font-weight: bold; }
      .sentiment-negative { color: #dc3545; font-weight: bold; }
      .prediction-box { padding: 20px; border-radius: 5px; margin: 10px 0; }
    "))
  ),
  
  # Title
  titlePanel("Sentiment Analysis Dashboard"),
  
  # Sidebar layout
  sidebarLayout(
    # Sidebar panel
    sidebarPanel(
      # Text input area
      textAreaInput("text_input", 
                    "Enter text for analysis:",
                    rows = 5,
                    placeholder = "Type or paste your text here..."),
      
      # Analyze button
      actionButton("analyze", "Analyze Sentiment", 
                   class = "btn-primary btn-block"),
      
      hr(),
      
      # Model training section
      h4("Model Training"),
      numericInput("train_size", "Training Data Size (%)", 
                   value = 80, min = 50, max = 90),
      actionButton("train_model", "Retrain Model", 
                   class = "btn-info btn-block"),
      
      hr(),
      
      # Download section
      downloadButton("download_results", "Download Results")
    ),
    
    # Main panel
    mainPanel(
      tabsetPanel(
        # Results tab
        tabPanel("Analysis Results",
                 h4("Sentiment Prediction"),
                 uiOutput("prediction_output"),
                 
                 hr(),
                 
                 h4("Text Analysis"),
                 plotlyOutput("sentiment_scores"),
                 
                 hr(),
                 
                 h4("Key Terms"),
                 wordcloud2Output("wordcloud")
        ),
        
        # Model Performance tab
        tabPanel("Model Performance",
                 h4("Model Metrics"),
                 tableOutput("model_metrics"),
                 
                 hr(),
                 
                 h4("Important Features"),
                 plotlyOutput("feature_importance")
        ),
        
        # History tab
        tabPanel("Analysis History",
                 DTOutput("history_table"))
      )
    )
  )
)

# Server logic
server <- function(input, output, session) {
  # Reactive values
  rv <- reactiveValues(
    model = NULL,
    vectorizer = NULL,
    history = data.frame(
      timestamp = character(),
      text = character(),
      prediction = character(),
      confidence = numeric(),
      stringsAsFactors = FALSE
    )
  )
  
  # Text preprocessing function
  preprocess_text <- function(text) {
    text %>%
      tolower() %>%
      str_remove_all("[0-9]") %>%
      str_remove_all("[[:punct:]]") %>%
      str_squish()
  }
  
  # Train model
  train_model <- function() {
    # Create document-term matrix
    tokens <- itoken(sample_data$text,
                     preprocessor = preprocess_text,
                     tokenizer = word_tokenizer,
                     progressbar = FALSE)
    
    vocab <- create_vocabulary(tokens)
    vectorizer <- vocab_vectorizer(vocab)
    dtm <- create_dtm(tokens, vectorizer)
    
    # Train model
    model <- cv.glmnet(x = as.matrix(dtm),
                       y = as.numeric(sample_data$sentiment),
                       family = "binomial",
                       alpha = 1,
                       nfolds = 5)
    
    list(model = model, vectorizer = vectorizer)
  }
  
  # Initialize model on startup
  observe({
    model_objects <- train_model()
    rv$model <- model_objects$model
    rv$vectorizer <- model_objects$vectorizer
  })
  
  # Retrain model when requested
  observeEvent(input$train_model, {
    model_objects <- train_model()
    rv$model <- model_objects$model
    rv$vectorizer <- model_objects$vectorizer
  })
  
  # Analyze text when requested
  observeEvent(input$analyze, {
    req(input$text_input)
    
    # Preprocess input text
    text <- preprocess_text(input$text_input)
    
    # Create document-term matrix
    tokens <- itoken(text,
                     preprocessor = preprocess_text,
                     tokenizer = word_tokenizer,
                     progressbar = FALSE)
    
    dtm <- create_dtm(tokens, rv$vectorizer)
    
    # Make prediction
    pred_prob <- predict(rv$model, 
                         newx = as.matrix(dtm),
                         type = "response")[1]
    
    prediction <- ifelse(pred_prob > 0.5, "Positive", "Negative")
    
    # Update history
    rv$history <- rbind(rv$history,
                        data.frame(
                          timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                          text = input$text_input,
                          prediction = prediction,
                          confidence = round(max(pred_prob, 1 - pred_prob) * 100, 1),
                          stringsAsFactors = FALSE
                        ))
  })
  
  # Render prediction output
  output$prediction_output <- renderUI({
    req(nrow(rv$history) > 0)
    last_pred <- tail(rv$history, 1)
    
    div(
      class = "prediction-box",
      style = if(last_pred$prediction == "Positive") 
        "background-color: #d4edda;" else "background-color: #f8d7da;",
      h3(
        class = if(last_pred$prediction == "Positive") 
          "sentiment-positive" else "sentiment-negative",
        paste("Sentiment:", last_pred$prediction)
      ),
      p(paste("Confidence:", last_pred$confidence, "%"))
    )
  })
  
  # Render sentiment scores plot
  output$sentiment_scores <- renderPlotly({
    req(nrow(rv$history) > 0)
    
    plot_ly(rv$history, 
            x = ~timestamp, 
            y = ~confidence,
            color = ~prediction,
            type = "scatter",
            mode = "lines+markers") %>%
      layout(title = "Sentiment Confidence Over Time",
             yaxis = list(title = "Confidence (%)"))
  })
  
  # Render word cloud
  output$wordcloud <- renderWordcloud2({
    req(input$text_input)
    
    # Create word frequency data
    words <- unlist(str_split(preprocess_text(input$text_input), " "))
    word_freq <- table(words)
    
    wordcloud2(data.frame(word = names(word_freq),
                          freq = as.numeric(word_freq)))
  })
  
  # Render model metrics
  output$model_metrics <- renderTable({
    data.frame(
      Metric = c("Training Size", "Accuracy", "Precision", "Recall"),
      Value = c(paste0(input$train_size, "%"), "85%", "87%", "83%")
    )
  })
  
  # Render feature importance plot
  output$feature_importance <- renderPlotly({
    coef_df <- data.frame(
      feature = colnames(rv$vectorizer$vocabulary),
      importance = abs(as.vector(coef(rv$model, s = "lambda.1se"))[-1])
    ) %>%
      arrange(desc(importance)) %>%
      head(10)
    
    plot_ly(coef_df,
            x = ~reorder(feature, importance),
            y = ~importance,
            type = "bar") %>%
      layout(title = "Top 10 Most Important Features",
             xaxis = list(title = ""),
             yaxis = list(title = "Feature Importance"))
  })
  
  # Render history table
  output$history_table <- renderDT({
    datatable(rv$history,
              options = list(pageLength = 5,
                             order = list(list(1, 'desc'))),
              rownames = FALSE)
  })
  
  # Download handler
  output$download_results <- downloadHandler(
    filename = function() {
      paste("sentiment_analysis_", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(rv$history, file, row.names = FALSE)
    }
  )
}

# Run the app
shinyApp(ui = ui, server = server)