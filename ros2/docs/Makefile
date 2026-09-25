PDF = standalone.pdf

all: $(PDF)

$(PDF): standalone.tex ros2-chapter/chapter.tex ros2-chapter/chapter-preamble.tex \
        ros2-chapter/refs.bib $(wildcard ros2-chapter/figures/*.tex) \
        $(wildcard ros2-chapter/listings/*)
	latexmk -pdf -interaction=nonstopmode standalone.tex

clean:
	latexmk -C standalone.tex
	rm -f standalone.bbl

.PHONY: all clean
