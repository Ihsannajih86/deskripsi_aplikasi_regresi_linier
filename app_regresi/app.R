# Load library yang dibutuhkan
library(shiny)      # Untuk membuat aplikasi web interaktif
library(ggplot2)     # Untuk visualisasi data
library(readr)       # Untuk membaca file CSV
library(dplyr)       # Untuk manipulasi data
library(caret)       # Untuk machine learning (tidak dipakai langsung tapi mendukung pipeline)
library(corrplot)    # Untuk membuat correlation matrix
library(tidyr)       # Untuk fungsi manipulasi data seperti drop_na
library(broom)       # Untuk merapikan output model agar mudah dibaca

# Bagian UI dari aplikasi
ui <- fluidPage(
  titlePanel("Aplikasi Prediksi Regresi Linier Variabel Y"),  # Judul aplikasi
  
  sidebarLayout(
    sidebarPanel(
      # Input file untuk dataset training (format .csv)
      fileInput("data", "Unggah Dataset Training (.csv)", accept = ".csv"),
      
      # UI dinamis untuk memilih variabel X (fitur independen), Y (target), dan warna
      uiOutput("xvar"),
      uiOutput("yvar"),
      uiOutput("colorvar"),
      
      # Tombol untuk melatih model regresi
      actionButton("train", "Latih Model"),
      
      # Input file untuk dataset testing (data baru untuk diprediksi)
      fileInput("newdata", "Unggah Dataset Testing (.csv)", accept = ".csv"),
      
      # Tombol untuk memicu proses prediksi
      actionButton("predict", "Prediksi Data Baru"),
      
      # Input file model yang sudah disimpan sebelumnya (.rds)
      fileInput("load_model", "Muat Model (.rds)", accept = ".rds"),
      
      # Tombol untuk menyimpan model yang sudah dilatih ke file .rds
      downloadButton("savemodel", "Download Model (.rds)")
    ),
    
    # Tampilan utama terdiri dari beberapa tab:
    mainPanel(
      tabsetPanel(
        tabPanel("Data Preview", tableOutput("datatable"), verbatimTextOutput("summary")),  # Pratinjau dan ringkasan data
        tabPanel("Correlation Matrix", plotOutput("corrplot")),  # Visualisasi korelasi antar variabel numerik
        tabPanel("Exploratory Analysis", plotOutput("scatterplot")),  # Visualisasi awal (scatterplot)
        tabPanel("Model Regresi", verbatimTextOutput("modelsummary"), plotOutput("modelplot")),  # Hasil model dan visualisasi
        tabPanel("Prediksi Data Baru", tableOutput("prediksi"))  # Output prediksi dari data baru
      )
    )
  )
)

# Bagian server dari aplikasi
server <- function(input, output, session) {
  
  # Membaca data training dari input
  data <- reactive({
    req(input$data)  # Tunggu sampai file diunggah
    read_csv(input$data$datapath)  # Membaca file CSV ke dalam data.frame
  })
  
  # Menampilkan preview 6 baris awal dari dataset
  output$datatable <- renderTable({
    head(data())
  })
  
  # Menampilkan statistik ringkasan dataset
  output$summary <- renderPrint({
    summary(data())
  })
  
  # UI dinamis: pilihan variabel X
  output$xvar <- renderUI({
    req(data())
    selectInput("x", "Pilih Variabel X", choices = names(data()), multiple = TRUE)
  })
  
  # UI dinamis: pilihan variabel Y (target)
  output$yvar <- renderUI({
    req(data())
    selectInput("y", "Pilih Variabel Y", choices = names(data()))
  })
  
  # UI dinamis: pilihan variabel warna (opsional, untuk visualisasi)
  output$colorvar <- renderUI({
    req(data())
    selectInput("color", "Variabel Warna (opsional)", choices = names(data()))
  })
  
  # Visualisasi korelasi antar variabel numerik dalam bentuk matriks
  output$corrplot <- renderPlot({
    df <- data() %>% select_if(is.numeric)  # Filter hanya variabel numerik
    corr <- cor(df)  # Hitung korelasi antar kolom
    corrplot(corr, method = "color", addCoef.col = "black")  # Plot dengan nilai korelasi
  })
  
  # Visualisasi scatterplot antar variabel terpilih
  output$scatterplot <- renderPlot({
    req(input$x, input$y, input$color)  # Pastikan semua input tersedia
    ggplot(data(), aes_string(x = input$x[1], y = input$y, color = input$color)) +
      geom_point(size = 3) +
      theme_minimal()
  })
  
  # Variabel reaktif untuk menyimpan model regresi
  model <- reactiveVal()
  
  # Latih model regresi linier saat tombol "Latih Model" ditekan
  observeEvent(input$train, {
    req(input$x, input$y)  # Pastikan input tersedia
    df <- data() %>% drop_na(all_of(c(input$y, input$x)))  # Hapus baris dengan nilai NA
    formula <- as.formula(paste(input$y, "~", paste(input$x, collapse = "+")))  # Bentuk formula regresi
    trained <- lm(formula, data = df)  # Latih model linier
    model(trained)  # Simpan model ke variabel reaktif
    saveRDS(trained, "model_simpan.rds")  # Simpan model ke file lokal (sementara)
  })
  
  # Memuat model dari file .rds
  observeEvent(input$load_model, {
    req(input$load_model)
    mod <- readRDS(input$load_model$datapath)  # Baca file model
    model(mod)  # Simpan ke variabel reaktif
  })
  
  # Tampilkan ringkasan model (summary regresi)
  output$modelsummary <- renderPrint({
    req(model())
    summary(model())
  })
  
  # Plot aktual vs prediksi untuk data training
  output$modelplot <- renderPlot({
    req(model())
    df <- data()
    df$pred <- predict(model(), newdata = df)  # Tambahkan kolom prediksi
    ggplot(df, aes_string(x = input$y, y = "pred")) +
      geom_point(color = "blue") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
      labs(title = "Plot Aktual vs Prediksi", x = "Aktual", y = "Prediksi")
  })
  
  # Membuat prediksi pada dataset baru saat tombol "Prediksi Data Baru" ditekan
  prediksi_data <- eventReactive(input$predict, {
    req(input$newdata)
    new_df <- read_csv(input$newdata$datapath)  # Membaca file data testing
    mod <- model()  # Ambil model yang sudah dilatih
    new_df$Prediksi_Y <- predict(mod, newdata = new_df)  # Tambahkan hasil prediksi
    new_df
  })
  
  # Menampilkan hasil prediksi data baru (preview)
  output$prediksi <- renderTable({
    req(prediksi_data())
    head(prediksi_data())
  })
  
  # Handler untuk menyimpan model sebagai file .rds yang dapat diunduh
  output$savemodel <- downloadHandler(
    filename = function() {
      paste("model_regresi_", Sys.Date(), ".rds", sep = "")  # Nama file
    },
    content = function(file) {
      saveRDS(model(), file)  # Simpan model ke file .rds untuk diunduh
    }
  )
}

# Jalankan aplikasi shiny
shinyApp(ui, server)
