Imports System.Windows.Threading

' https://resanaplaza.com/2023/06/24/%E3%80%90%E3%82%B5%E3%83%B3%E3%83%97%E3%83%AB%E6%BA%80%E8%BC%89%E3%80%91c%E3%81%A7%E5%8B%95%E7%94%BB%E5%86%8D%E7%94%9F%E3%81%97%E3%82%88%E3%81%86%E3%82%88%EF%BC%81%EF%BC%88mediaelement%EF%BC%89/

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


    Private Sub LoadMovieButton_Click(sender As Object, e As RoutedEventArgs) Handles LoadMovieButton.Click
        Dim a = New Microsoft.Win32.OpenFileDialog()
        Dim result = a.ShowDialog()
        If result Then
            moviepath = a.FileName
            mediaElement.Source = New Uri(moviepath)
            slider.Value = 0
            mediaElement.Play()
            mediaElement.Stop()
            Do Until mediaElement.NaturalDuration.HasTimeSpan ' 再生中でないとこれがTrueにならない

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



        Dim foo = IO.Path.GetDirectoryName(Environment.CurrentDirectory & "\" & pyexepath)
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

    Private Sub Slider_ValueChanged(sender As Object, e As RoutedPropertyChangedEventArgs(Of Double)) Handles slider.ValueChanged
        'If sliderdrag Then
        '    SetTimeLabel()
        'End If
    End Sub

    Private Sub Ps_Exited(sender As Object, e As EventArgs)
        'プロセスが終了したときに実行される
        ps_test_running = False
        Dim ps = DirectCast(sender, Process)
        If ps IsNot Nothing Then
            ps.Close()
            ps.Dispose()
        End If
        MessageBox.Show("finished")
        Dispatcher.Invoke(
            Sub()
                SaveButton.IsEnabled = True
            End Sub
        )
    End Sub

    Private Sub Ps_OutputDataReceived(sender As Object, e As DataReceivedEventArgs)
        Dispatcher.Invoke(
            Sub()
                OutputTextBox.Text += e.Data & vbCrLf
                OutputTextBox.ScrollToEnd()
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
        StartPyProcess(ps_test, "run_eval.py " & moviepath, ".\..\..\..\..\..\")
        ps_test_running = True
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

    Private Sub ZeroButton_Click(sender As Object, e As RoutedEventArgs) Handles ZeroButton.Click
        mediaElement.Position = TimeSpan.FromMilliseconds(0)
    End Sub

    Private Sub ChangeControlIsEnabled(val As Boolean)
        slider.IsEnabled = val
        ZeroButton.IsEnabled = val
        PlayButton.IsEnabled = val
        PauseButton.IsEnabled = val
        StopButton.IsEnabled = val
        EvalButton.IsEnabled = val
    End Sub

    Private Sub SaveButton_Click(sender As Object, e As RoutedEventArgs) Handles SaveButton.Click
        Dim a = New Microsoft.Win32.SaveFileDialog()
        Dim result = a.ShowDialog()
        If result Then

        End If
    End Sub
End Class
