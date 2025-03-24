
/* Hide author sidebar profile only, not the TOC sidebar */
.author__avatar,
.author__content,
.author__urls-wrapper {
  display: none !important;
}

/* Make main content match header width */
#main {
  max-width: 100%;
  width: 100%;
  padding-left: 1em;
  padding-right: 1em;

  @include breakpoint($x-large) {
    max-width: $max-width;
    margin: 0 auto;
  }
}

/* Fix layout for all content types */
.archive,
.page {
  @include breakpoint($large) {
    width: 100%;
    padding-right: 0;
  }

  @include breakpoint($x-large) {
    width: 100%;
    padding-right: 0;
  }
}

/* Direct fix for article/page content */
article.page {
  float: none;
  width: 100% !important;
  max-width: $max-width !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

/* TOC styling fixes */
.sidebar__right {
  display: block !important;
  position: relative !important;
  margin-right: 0 !important;
  padding-left: 0 !important;
  clear: both;
  margin-bottom: 2em;
  float: none !important;
  width: 100% !important;
}

/* Fix header styling */
.toc .nav__title {
  color: #fff;
  background-color: $primary-color;
  width: 100%;
  display: block;
  padding: 0.5rem 0.75rem;
  margin: 0;
  border-top-left-radius: $border-radius;
  border-top-right-radius: $border-radius;
}

/* Proper TOC styling */
.toc {
  font-family: $sans-serif-narrow;
  color: $gray;
  background-color: $background-color;
  border: 1px solid $border-color;
  border-radius: $border-radius;
  box-shadow: $box-shadow;
  margin-bottom: 2em;
}

.toc__menu {
  margin: 0;
  padding: 0;
  width: 100%;
  list-style: none;
  font-size: $type-size-6;
  display: block; /* Force vertical layout */

  a {
    display: block;
    padding: 0.5rem 0.75rem;
    color: $muted-text-color;
    font-weight: bold;
    line-height: 1.5;
    border-bottom: 1px solid $border-color;

    &:hover {
      color: $text-color;
      background-color: mix(#fff, $background-color, 10%);
    }
  }

  li ul > li a {
    padding-left: 1.25rem;
    font-weight: normal;
  }

  li ul li ul > li a {
    padding-left: 1.75rem;
  }
}

/* Hide pagination if needed */
.pagination {
  display: none;
}