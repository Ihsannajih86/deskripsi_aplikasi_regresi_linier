# ======================================
# Aplikasi R Shiny: Analisis Regresi Linier
# Fitur: Upload Data, Korelasi, Scatter Plot, Model Regresi, Prediksi
# ======================================

# --- Library yang digunakan ---
library(shiny)           # Untuk membuat aplikasi web interaktif
library(tidyverse)       # Untuk manipulasi data dan visualisasi
library(DT)              # Menampilkan tabel interaktif
library(ggplot2)         # Grafik visualisasi
library(corrplot)        # Visualisasi korelasi variabel
library(shinythemes)     # Tema tampilan aplikasi
library(broom)           # Ringkasan hasil model
library(readr)           # Membaca file CSV

# --- UI Aplikasi ---
ui <- fluidPage(
  theme = shinytheme("cosmo"),  # Tema aplikasi
  titlePanel("Aplikasi Analisis Regresi Linier"),
  sidebarLayout(
    sidebarPanel(
      fileInput("train_data", "Unggah Dataset Training (.csv)", accept = ".csv"),
      fileInput("test_data", "Unggah Dataset Testing (.csv)", accept = ".csv"),
      uiOutput("var_select_ui"),  # Dropdown untuk memilih variabel X dan Y
      actionButton("save_model", "Simpan Model"),
      fileInput("load_model", "Muat Model (.rds)", accept = ".rds")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Data Preview",
                 DTOutput("preview_table"),
                 verbatimTextOutput("summary")
        ),
        tabPanel("Correlation Matrix",
                 plotOutput("cor_plot")
        ),
        tabPanel("Exploratory Analysis",
                 plotOutput("scatter_plot")
        ),
        tabPanel("Model Regresi",
                 actionButton("build_model", "Bangun Model"),
                 verbatimTextOutput("model_summary"),
                 verbatimTextOutput("model_metrics"),
                 plotOutput("actual_vs_pred")
        ),
        tabPanel("Prediksi Data Baru",
                 DTOutput("predicted_table")
        )
      )
    )
  )
)

# --- Server Aplikasi ---
server <- function(input, output, session) {
  # Membaca dataset training
  train_data <- reactive({
    req(input$train_data)
    read_csv(input$train_data$datapath)
  })
  
  # Membaca dataset testing
  test_data <- reactive({
    req(input$test_data)
    read_csv(input$test_data$datapath)
  })
  
  # Dropdown pilihan variabel numerik X dan Y
  output$var_select_ui <- renderUI({
    req(train_data())
    data <- train_data()
    numeric_vars <- names(data)[sapply(data, is.numeric)]
    tagList(
      selectInput("x_var", "Pilih Variabel X:", choices = numeric_vars),
      selectInput("y_var", "Pilih Variabel Y:", choices = numeric_vars)
    )
  })
  
  # Preview data
  output$preview_table <- renderDT({
    req(train_data())
    datatable(head(train_data(), 20))
  })
  
  # Ringkasan statistik
  output$summary <- renderPrint({
    req(train_data())
    summary(train_data())
  })
  
  # Matriks korelasi antar variabel numerik
  output$cor_plot <- renderPlot({
    req(train_data())
    cor_matrix <- cor(train_data()[, sapply(train_data(), is.numeric)], use = "complete.obs")
    corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", addCoef.col = "black")
  })
  
  # Scatter plot antara X dan Y
  output$scatter_plot <- renderPlot({
    req(train_data(), input$x_var, input$y_var)
    ggplot(train_data(), aes_string(x = input$x_var, y = input$y_var, color = input$y_var)) +
      geom_point() +
      scale_color_gradient(low = "blue", high = "red") +
      theme_minimal()
  })
  
  # Variabel reaktif untuk menyimpan model
  model <- reactiveVal(NULL)
  
  # Membuat model regresi linier
  observeEvent(input$build_model, {
    req(train_data(), input$x_var, input$y_var)
    formula <- as.formula(paste(input$y_var, "~", input$x_var))
    model(lm(formula, data = train_data()))
  })
  
  # Menyimpan model ke file
  observeEvent(input$save_model, {
    req(model())
    saveRDS(model(), file = "saved_model.rds")
  })
  
  # Memuat model dari file
  observeEvent(input$load_model, {
    req(input$load_model)
    loaded <- readRDS(input$load_model$datapath)
    model(loaded)
  })
  
  # Ringkasan model
  output$model_summary <- renderPrint({
    req(model())
    summary(model())
  })
  
  # Metrik model (R-squared, p-value, dll)
  output$model_metrics <- renderPrint({
    req(model())
    glance(model())
  })
  
  # Grafik prediksi vs aktual
  output$actual_vs_pred <- renderPlot({
    req(model())
    pred <- predict(model(), newdata = train_data())
    actual <- train_data()[[input$y_var]]
    ggplot(data.frame(Actual = actual, Predicted = pred), aes(x = Actual, y = Predicted)) +
      geom_point(color = "darkgreen") +
      geom_smooth(method = "lm", se = FALSE, color = "red") +
      theme_minimal()
  })
  
  # Prediksi data testing
  output$predicted_table <- renderDT({
    req(model(), test_data())
    tryCatch({
      preds <- predict(model(), newdata = test_data())
      result <- test_data()
      result$Predicted_Y <- preds
      datatable(result)
    }, error = function(e) {
      datatable(data.frame(Error = "Gagal melakukan prediksi. Pastikan struktur data testing sesuai."))
    })
  })
}

# --- Jalankan Aplikasi ---
shinyApp(ui = ui, server = server)