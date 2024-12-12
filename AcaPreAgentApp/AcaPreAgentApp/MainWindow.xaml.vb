Imports System.Windows.Threading

' https://resanaplaza.com/2023/06/24/%E3%80%90%E3%82%B5%E3%83%B3%E3%83%97%E3%83%AB%E6%BA%80%E8%BC%89%E3%80%91c%E3%81%A7%E5%8B%95%E7%94%BB%E5%86%8D%E7%94%9F%E3%81%97%E3%82%88%E3%81%86%E3%82%88%EF%BC%81%EF%BC%88mediaelement%EF%BC%89/

' https://vdlz.xyz/Csharp/Porpose/Editor/RichTextBoxEditor/RichTextBoxEditor09_ProgrammaticallyInput.html

Class MainWindow

    Private moviepath As String
    Private Timer1 As Threading.DispatcherTimer
    Private pyexepath = "..\..\..\..\..\.venv\Scripts\python.exe"
    Private pypath = "..\..\..\..\..\test.py"
    Private videodemopath = "..\..\..\..\..\video_demo.py"
    Private apidemopath = "..\..\..\..\..\api_demo.py"
    Private rootpath = Environment.CurrentDirectory

    Private ps_test As Process, ps_test_running As Boolean

    Private sliderdrag As Boolean

    Private pagetime As New List(Of Double)

    Private IsSuggestionStarted As Boolean


    Private Sub LoadMovieButton_Click(sender As Object, e As RoutedEventArgs) Handles LoadMovieButton.Click
        Dim a = New Microsoft.Win32.OpenFileDialog()
        Dim result = a.ShowDialog()
        If result Then
            moviepath = a.FileName
            mediaElement.Source = New Uri(moviepath)
            slider.Value = 0
            mediaElement.Play()
            mediaElement.Stop()
            Do Until mediaElement.NaturalDuration.HasTimeSpan ' 動画を読み込んでいないとこれがTrueにならない

            Loop
            slider.Maximum = mediaElement.NaturalDuration.TimeSpan.TotalMilliseconds
            SetTimeLabel()
            ChangeControlIsEnabled(True)
        End If
    End Sub

    Private Sub PlayButton_Click(sender As Object, e As RoutedEventArgs) Handles PlayButton.Click
        mediaElement.Play()
    End Sub

    Private Sub StopButton_Click(sender As Object, e As RoutedEventArgs) Handles StopButton.Click
        mediaElement.Stop()
    End Sub

    Private Sub PauseButton_Click(sender As Object, e As RoutedEventArgs) Handles PauseButton.Click
        mediaElement.Pause()
    End Sub

    Private Sub MainWindow_Loaded(sender As Object, e As RoutedEventArgs) Handles Me.Loaded
        Console.WriteLine(Environment.CurrentDirectory)
        ChangeControlIsEnabled(False)
        SaveButton.IsEnabled = False
        slider.Minimum = 0
        slider.Maximum = 10000
        slider.Value = 0
        sliderdrag = False

        'Dim fd As New FlowDocument
        'Dim par As New Paragraph
        'Dim h As New Hyperlink
        'Dim r As New Run

        'r.Text = "test link 2"
        'h.Name = "testlink2"
        'h.NavigateUri = New Uri("https://example.com")
        'AddHandler h.RequestNavigate, AddressOf Hyperlink_RequestNavigate
        'h.Inlines.Add(r)
        'par.Inlines.Add(h)
        'fd.Blocks.Add(par)
        'richTextBox.Document = fd

        Timer1 = New Threading.DispatcherTimer With {
            .Interval = New TimeSpan(1000000) ' 100万=100ms
            }
        AddHandler Timer1.Tick, New EventHandler(AddressOf Timer1_Tick)
        Timer1.Start()
    End Sub

    Private Sub Timer1_Tick(sender As Object, e As EventArgs)
        If Not sliderdrag Then slider.Value = mediaElement.Position.TotalMilliseconds
        SetTimeLabel()
    End Sub

    Private Sub Ps_Exited(sender As Object, e As EventArgs)
        'プロセスが終了したときに実行される
        ps_test_running = False
        Dim ps = DirectCast(sender, Process)
        If ps IsNot Nothing Then
            ps.WaitForExit()
            ps.Close()
            ps.Dispose()
        End If
        Dispatcher.Invoke(
            Sub()
                SaveButton.IsEnabled = True
                EvalButton.IsEnabled = True
                LoadMovieButton.IsEnabled = True
                EvalButton.Content = "Run Evaluation"
                Pb1.IsIndeterminate = False

                If pagetime.Count > 0 Then
                    ' pagetimeの内容を反映
                    Dim fd As New FlowDocument
                    Dim par As New Paragraph

                    For i As Integer = 0 To pagetime.Count - 2
                        Dim ts = TimeSpan.FromSeconds(pagetime(i))
                        Dim h As New Hyperlink
                        Dim r As New Run With {
                            .Text = "page" & (i + 1).ToString & " (" & ts.ToString("hh\:mm\:ss") & ")"
                        }
                        h.Name = "page" & (i + 1).ToString
                        h.Tag = pagetime(i)
                        h.NavigateUri = New Uri("https://example.com/")
                        AddHandler h.RequestNavigate, AddressOf Hyperlink_RequestNavigate
                        h.Inlines.Add(r)
                        par.Inlines.Add(h)

                        Dim r2 As New Run With {
                            .Text = " "
                        }
                        par.Inlines.Add(r2)
                    Next
                    fd.Blocks.Add(par)
                    richTextBox.Document = fd
                End If


            End Sub
        )
        MessageBox.Show("finished")
    End Sub

    Private Sub Ps_OutputDataReceived(sender As Object, e As DataReceivedEventArgs)
        ' スライド切り替わり時間のフォーマットは Time stamps: [0.0, 15.999831223628693, 39.199586497890294, 56.99939873417722, 77.59918143459916]
        Dispatcher.Invoke(
            Sub()
                OutputTextBox.Text += e.Data & vbCrLf
                OutputTextBox.ScrollToEnd()
                If IsSuggestionStarted Then
                    SuggestionTextBox.Text += e.Data & vbCrLf
                ElseIf e.Data.StartsWith("Time stamps: [") Then
                    pagetime.Clear()
                    Dim s = e.Data.Substring(14, e.Data.Length - 15).Split(","c)
                    For i As Integer = 0 To s.Length - 1
                        pagetime.Add(CDbl(s(i)))
                    Next
                ElseIf e.Data.StartsWith("slide number ") Then
                    TranscriptTextBox.Text += e.Data & vbCrLf
                ElseIf e.Data.StartsWith("AI Suggestion Result:") Then
                    IsSuggestionStarted = True
                End If
            End Sub
        )
    End Sub

    Private Sub Ps_ErrorDataReceived(sender As Object, e As DataReceivedEventArgs)
        Dispatcher.Invoke(
            Sub()
                OutputTextBox.Text += e.Data & vbCrLf
                OutputTextBox.ScrollToEnd()
            End Sub
        )
    End Sub

    Private Sub EvalButton_Click(sender As Object, e As RoutedEventArgs) Handles EvalButton.Click
        ' https://qiita.com/kktkhs1936/items/a3ea2a25d1c91fff1f52
        'StartPyProcess(ps_test, "test.py videos\confident.mp4", ".\..\..\..\..\..\")
        'StartPyProcess(ps_test, "run_eval.py videos\confident.mp4", ".\..\..\..\..\..\")
        StartPyProcess(ps_test, "run_eval_wpf.py " & moviepath, ".\..\..\..\..\..\")
        ps_test_running = True
        EvalButton.IsEnabled = False
        LoadMovieButton.IsEnabled = False
        Pb1.IsIndeterminate = True
        EvalButton.Content = "Evaluating..."
        OutputTextBox.Clear()
        TranscriptTextBox.Clear()
        SuggestionTextBox.Clear()
        richTextBox.Document = New FlowDocument()
        IsSuggestionStarted = False
    End Sub

    Private Sub StartPyProcess(ps As Process, filename As String, wdir As String)
        ps = New Process()
        With ps.StartInfo
            .FileName = pyexepath
            .Arguments = "-u " & filename
            .WorkingDirectory = wdir ' ".\..\..\..\..\..\"
            .RedirectStandardOutput = True
            .RedirectStandardError = True
            .UseShellExecute = False
            .CreateNoWindow = True
        End With
        ps.EnableRaisingEvents = True
        AddHandler ps.OutputDataReceived, AddressOf Ps_OutputDataReceived
        AddHandler ps.ErrorDataReceived, AddressOf Ps_ErrorDataReceived
        AddHandler ps.Exited, AddressOf Ps_Exited
        ps.Start()
        ps.BeginOutputReadLine()
        ps.BeginErrorReadLine()
    End Sub

    Private Sub slider_PreviewMouseDown(sender As Object, e As MouseButtonEventArgs) Handles slider.PreviewMouseDown
        sliderdrag = True
    End Sub

    Private Sub slider_PreviewMouseUp(sender As Object, e As MouseButtonEventArgs) Handles slider.PreviewMouseUp
        sliderdrag = False
        mediaElement.Position = TimeSpan.FromMilliseconds(slider.Value)
    End Sub

    Private Sub SetTimeLabel()
        If mediaElement.NaturalDuration.HasTimeSpan Then
            TimeLabel.Content = TimeSpan.FromMilliseconds(slider.Value).ToString("hh\:mm\:ss") & " / " & mediaElement.NaturalDuration.TimeSpan.ToString("hh\:mm\:ss")
        Else
            TimeLabel.Content = "00:00:00 / 00:00:00"
        End If
    End Sub

    'Private Sub ZeroButton_Click(sender As Object, e As RoutedEventArgs) Handles ZeroButton.Click
    '    mediaElement.Position = TimeSpan.FromMilliseconds(0)
    'End Sub

    Private Sub ChangeControlIsEnabled(val As Boolean)
        slider.IsEnabled = val
        'ZeroButton.IsEnabled = val
        PlayButton.IsEnabled = val
        PauseButton.IsEnabled = val
        StopButton.IsEnabled = val
        EvalButton.IsEnabled = val
    End Sub

    Private Sub Hyperlink_RequestNavigate(sender As Object, e As RequestNavigateEventArgs)
        Dim h = DirectCast(sender, Hyperlink)
        Try
            Dim t = CDbl(h.Tag)
            mediaElement.Position = TimeSpan.FromSeconds(t)
        Catch ex As Exception

        End Try
        'MessageBox.Show(h.Name & vbCrLf & pagetime.Count)
    End Sub

    Private Sub SaveButton_Click(sender As Object, e As RoutedEventArgs) Handles SaveButton.Click
        Dim a = New Microsoft.Win32.SaveFileDialog()
        a.Filter = "Text Files (*.txt)|*.txt|All Files (*.*)|*.*"
        Dim result = a.ShowDialog()
        If result Then
            Using w As New IO.StreamWriter(a.FileName, False, Text.Encoding.UTF8)
                w.Write(SuggestionTextBox.Text)
            End Using
        End If
    End Sub


End Class
