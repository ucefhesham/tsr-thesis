$view                    = 'pdf';
$out_dir                 = 'out';
$clean_ext               = 'run.xml bbl';
$pdflatex                = 'xelatex -file-line-error -halt-on-error %O %S';

$pdf_mode                = 1;
$postscript_mode         = 0;
$dvi_mode                = 0;

$preview_continuous_mode = 1;  # equivalent to -pvc
$cleanup_mode            = 2;  # everything but dvi/ps/pdf
