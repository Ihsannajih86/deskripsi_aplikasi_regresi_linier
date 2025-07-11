library(shiny)
library(ggplot2)
library(readr)
library(dplyr)
library(caret)
library(corrplot)
library(tidyr)+
library(broom)

ui <- fluidPage(
  titlePanel("Aplikasi Prediksi Regresi Linier Variabel Y"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("data", "Unggah Dataset Training (.csv)", accept = ".csv"),
      uiOutput("xvar"),
      uiOutput("yvar"),
      uiOutput("colorvar"),
      actionButton("train", "Latih Model"),
      fileInput("newdata", "Unggah Dataset Testing (.csv)", accept = ".csv"),
      actionButton("predict", "Prediksi Data Baru"),
      fileInput("load_model", "Muat Model (.rds)", accept = ".rds"),
      downloadButton("savemodel", "Download Model (.rds)")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Data Preview", tableOutput("datatable"), verbatimTextOutput("summary")),
        tabPanel("Correlation Matrix", plotOutput("corrplot")),
        tabPanel("Exploratory Analysis", plotOutput("scatterplot")),
        tabPanel("Model Regresi", verbatimTextOutput("modelsummary"), plotOutput("modelplot")),
        tabPanel("Prediksi Data Baru", tableOutput("prediksi"))
      )
    )
  )
)

server <- function(input, output, session) {
  
  data <- reactive({
    req(input$data)
    read_csv(input$data$datapath)
  })
  
  output$datatable <- renderTable({
    head(data())
  })
  
  output$summary <- renderPrint({
    summary(data())
  })
  
  output$xvar <- renderUI({
    req(data())
    selectInput("x", "Pilih Variabel X", choices = names(data()), multiple = TRUE)
  })
  
  output$yvar <- renderUI({
    req(data())
    selectInput("y", "Pilih Variabel Y", choices = names(data()))
  })
  
  output$colorvar <- renderUI({
    req(data())
    selectInput("color", "Variabel Warna (opsional)", choices = names(data()))
  })
  
  output$corrplot <- renderPlot({
    df <- data() %>% select_if(is.numeric)
    corr <- cor(df)
    corrplot(corr, method = "color", addCoef.col = "black")
  })
  
  output$scatterplot <- renderPlot({
    req(input$x, input$y, input$color)
    ggplot(data(), aes_string(x = input$x[1], y = input$y, color = input$color)) +
      geom_point(size = 3) +
      theme_minimal()
  })
  
  model <- reactiveVal()
  
  observeEvent(input$train, {
    req(input$x, input$y)
    df <- data() %>% drop_na(all_of(c(input$y, input$x)))
    formula <- as.formula(paste(input$y, "~", paste(input$x, collapse = "+")))
    trained <- lm(formula, data = df)
    model(trained)
    saveRDS(trained, "model_simpan.rds")
  })
  
  observeEvent(input$load_model, {
    req(input$load_model)
    mod <- readRDS(input$load_model$datapath)
    model(mod)
  })
  
  output$modelsummary <- renderPrint({
    req(model())
    summary(model())
  })
  
  output$modelplot <- renderPlot({
    req(model())
    df <- data()
    df$pred <- predict(model(), newdata = df)
    ggplot(df, aes_string(x = input$y, y = "pred")) +
      geom_point(color = "blue") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
      labs(title = "Plot Aktual vs Prediksi", x = "Aktual", y = "Prediksi")
  })
  
  prediksi_data <- eventReactive(input$predict, {
    req(input$newdata)
    new_df <- read_csv(input$newdata$datapath)
    mod <- model()
    new_df$Prediksi_Y <- predict(mod, newdata = new_df)
    new_df
  })
  
  output$prediksi <- renderTable({
    req(prediksi_data())
    head(prediksi_data())
  })
  
  output$savemodel <- downloadHandler(
    filename = function() {
      paste("model_regresi_", Sys.Date(), ".rds", sep = "")
    },
    content = function(file) {
      saveRDS(model(), file)
    }
  )
}

shinyApp(ui, server)