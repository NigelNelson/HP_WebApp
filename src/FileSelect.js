class FileSelect extends React.Component {
    render() {
        return (
            <div className="mb-3">
                <label htmlFor="formFile" className="form-label">{this.props.promptStatement}</label>
                <input className="form-control"
                       type="file"
                       id="formFile"
                       onChange={(event) => this.props.onFileSelect(event.target.value)}
                       accept={this.props.acceptedFile}/>
            </div>
        )
    }
}