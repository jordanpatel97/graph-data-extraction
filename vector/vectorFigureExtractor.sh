#Shell Script to convert all PDF pages to SVG files
#Run using ./extract_pdf_images.sh "/home/ubuntu/Documents/vector_extraction/" "test.pdf"

!/bin/bash

TMP_DIR=$1
SOURCE_PDF=$2
MAX_WIDTH=1920
MAX_HEIGHT=1080

echo "$TMP_DIR"
echo "source: $SOURCE_PDF"

#Copy original PDF to same dir but a new name
cp $SOURCE_PDF $SOURCE_PDF'.ori'

function burst
{
    local source=$1

    # explodes the pages to pdf files
    `/usr/bin/pdftk $source burst`

    # removes the pdftk txt files 
    `rm *.txt`
}


function process_pages {
    local tmp_dir=$1
    local pnum=1

    for f in `find . -type f -name "*.pdf"`
    do
        echo "processing page $f"
        check_for_images $f $pnum
        let "pnum++"
    done
}


function check_for_images {
    local pdf_page=$1
    local pnum=$2

    # Assume it contains a vector image and extract vector images
    # ToDo make more efficient
    extract_vector_images $pdf_page $pnum

    rm $pdf_page
}




function extract_vector_images {
    local pdf_page=$1
    local pnum=$2

    pdf_file="${pdf_page%.*}"

    echo "extract vector image from the page as SVG"
    `/usr/bin/inkscape --without-gui --file=$pdf_page --export-plain-svg=$pdf_page.svg` #CREATES PLAIN SVG
    #`/usr/bin/inkscape --without-gui --file=$pdf_page --export-area=50:750:325:1000 --export-png=$pdf_page.png` #EXPORTS SPECIFIC AREA TO PNG
    #`/usr/bin/inkscape --without-gui --file=$pdf_page --export-eps=$pdf_page.eps` #CREATES EPS
}


cd $TMP_DIR
burst $SOURCE_PDF
process_pages $TMP_DIR
