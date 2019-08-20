<?php 
$st = $_REQUEST['string'];
$inp = $_REQUEST['input'];
$pyscript = 'C:\\xampp\\htdocs\\BEPROJ\\test.py';
$python = 'C:\\Users\\ranet\\AppData\\Local\\Programs\\Python\\Python37\\python.exe';
$cmd = "$python $pyscript $st $inp";
exec("$cmd",$output);
print_r($output);


?>