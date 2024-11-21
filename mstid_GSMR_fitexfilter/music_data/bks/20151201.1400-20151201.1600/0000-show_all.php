<?php
foreach (glob("*.png") as $filename) {
    echo "<img src='$filename' > <br />";
}
?>